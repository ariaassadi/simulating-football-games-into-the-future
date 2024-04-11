import argparse
import json
import time
import socket

import torch


class PitchControl():
    pitch_w: int | None = 68
    pitch_l: int | None = 105
    max_influence_distance: float | None = 20.0
    max_influence_radius: float | None = 15.0
    max_player_speed: float | None = 13.0
    max_player_influence: float | None = 0.73

    def _speed_vector(
        self, speed: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(
            [torch.cos(direction) * speed, torch.sin(direction) * speed]
        )

    def _rotation_matrix(self, direction: torch.Tensor) -> torch.Tensor:
        cos_theta, sin_theta = torch.cos(direction), torch.sin(direction)
        return torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta]).reshape(2, 2)

    def _scaling_matrix(
        self, speed: torch.Tensor, radius: torch.Tensor
    ) -> torch.Tensor:
        speed_ratio = (speed**2) / (self.max_player_speed**2)
        scaling_matrix = torch.zeros((2, 2))
        scaling_matrix[0, 0] = (radius + (radius * speed_ratio)) / 2
        scaling_matrix[1, 1] = (radius - (radius * speed_ratio)) / 2
        return scaling_matrix

    def _influence_radius(self, distance_to_ball: torch.Tensor) -> torch.Tensor:
        factor = 5.7155921353452216e-05  # 1/17496
        radius = (factor * distance_to_ball**4) + 4.0
        return torch.minimum(radius, torch.tensor(self.max_influence_radius))

    def _covariance_matrix(
        self,
        distance_to_ball: torch.Tensor,
        direction: torch.Tensor,
        speed: torch.Tensor,
    ) -> torch.Tensor:
        radius = self._influence_radius(distance_to_ball)
        s_matrix = self._scaling_matrix(speed, radius)
        r_matrix = self._rotation_matrix(direction)
        return r_matrix @ s_matrix @ s_matrix @ r_matrix.transpose(-2, -1)

    def _player_influence(
        self,
        *,
        grid: torch.Tensor,
        ball_pos: torch.Tensor,
        player_pos: torch.Tensor,
        speed: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        distance_to_ball = torch.linalg.norm(player_pos - ball_pos)
        cov = self._covariance_matrix(distance_to_ball, direction, speed)
        mean = player_pos + 0.5 * self._speed_vector(speed, direction)
        diff = grid - mean
        dist = self._mahalanobis_cholesky(diff, cov)
        return self._distribution(dist)

    def _mahalanobis_cholesky(
        self, diff: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        L = torch.linalg.cholesky(cov)
        diff = diff[:, None].transpose(-2, -1)
        y = torch.linalg.solve_triangular(L, diff, upper=False)
        return torch.einsum('ijk,ijk->i', y, y)

    def _distribution(self, distance: torch.Tensor) -> torch.Tensor:
        # We are doing f_pos / f_mean but note that since the distribution is calculated as
        # f(x) = exp(-0.5 * (x - mu)^T * Sigma^-1 * (x - mu)) / sqrt((2 * pi)^k * det(Sigma))
        # we can simplify the division to just the exponential part
        return torch.exp(-0.5 * distance)

    def _normalize_control(self, control: torch.Tensor) -> torch.Tensor:
        k = -torch.log(torch.tensor([(1 / self.max_player_influence) - 1]))
        return 1.0 / (1.0 + torch.exp(-control * k))

    def _influence_matrix(
        self,
        *,
        ball_pos: torch.Tensor,
        player_pos: torch.Tensor,
        speed: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        # grid_x, grid_y = torch.meshgrid(
        #     torch.linspace(-self.pitch_l / 2, self.pitch_l / 2, self.pitch_l),
        #     torch.linspace(self.pitch_w / 2, -self.pitch_w / 2, self.pitch_w),
        #     indexing='xy',
        # )
        
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-self.pitch_l / 2, self.pitch_l / 2, self.pitch_l),
            torch.linspace(self.pitch_w / 2, -self.pitch_w / 2, self.pitch_w),
            indexing='xy',
        )
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

        distances = torch.linalg.norm(grid - player_pos, dim=1)
        inside_grid = distances <= self.max_influence_distance

        influences = self._player_influence(
            grid=grid[inside_grid],
            ball_pos=ball_pos,
            player_pos=player_pos,
            speed=speed,
            direction=direction,
        )

        player_influence = torch.zeros(grid.shape[0])
        player_influence[inside_grid] = influences
        return player_influence.reshape(self.pitch_w, self.pitch_l)

    def pitch_control(self, ball_pos: torch.Tensor, players: list[dict]) -> torch.Tensor:
        influences = torch.zeros((self.pitch_w, self.pitch_l))

        for player in players:
            influence = self._influence_matrix(
                ball_pos=torch.tensor(
                    [ball_pos['x'], ball_pos['y']],
                    dtype=torch.float32,
                ),
                player_pos=torch.tensor(
                    [player['x'], player['y']], dtype=torch.float32
                ),
                speed=torch.tensor([player['speed']], dtype=torch.float32),
                direction=torch.tensor(
                    [player['direction']], dtype=torch.float32
                ),
            )

            influences += influence if player['team'] == 'home_team' else -influence

        return influences


def main():
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('localhost', 12345)  # Change the host and port as needed
    print('Starting up on {} port {}'.format(*server_address))
    server_socket.bind(server_address)

    # Listen for incoming connections
    server_socket.listen(1)

    while True:
        print('Waiting for a connection...')
        connection, client_address = server_socket.accept()

        try:
            print('Connection from', client_address)

            while True:
                data = connection.recv(8192)
                if data:
                    player_data = json.loads(data.decode()) # line 168
                    # print('Received data from the client: ', player_data)
                    ball = next(p for p in player_data['players'] if p['team'] == 'ball')
                    players = [p for p in player_data['players'] if p['team'] != 'ball']
                    for p in players:
                        p['speed'] = p['v']
                        p['direction'] = p['orientation']

                    pc = PitchControl()
                    pitch_control = pc.pitch_control(ball_pos=ball, players=players)
                    pitch_list = pitch_control.tolist()
                    flattened_list = [item for sublist in pitch_list for item in sublist]
                    format = {"pitch": flattened_list}

                    print('Size of the data to be sent to the client: ', len(json.dumps(format).encode()))
                    connection.sendall(json.dumps(format).encode())
                else:
                    print('No more data from', client_address)
                    break
        finally:
            # Clean up the connection
            connection.close()


if __name__ == "__main__":
    main() # line 193
    
# python pitch_control_main.py '{"players":[{"x":65.66999816894531,"y":30.8799991607666,"team":"home_team","jersey_number":6,"v":1.7677669525146484,"orientation":6.1412882804870605},{"x":71.29000091552734,"y":29.510000228881836,"team":"home_team","jersey_number":19,"v":3.4003677368164062,"orientation":5.341185092926025},{"x":77.37999725341797,"y":35.66999816894531,"team":"home_team","jersey_number":18,"v":3.6912057399749756,"orientation":5.789244174957275},{"x":74.5999984741211,"y":56.84000015258789,"team":"home_team","jersey_number":11,"v":1.8027756214141846,"orientation":6.871187686920166},{"x":63.619998931884766,"y":23.25,"team":"home_team","jersey_number":23,"v":4.100305080413818,"orientation":6.938880920410156},{"x":53.47999954223633,"y":32.90999984741211,"team":"home_team","jersey_number":2,"v":1.7677669525146484,"orientation":6.425082206726074},{"x":61.95000076293945,"y":39.540000915527344,"team":"home_team","jersey_number":10,"v":3.041381359100342,"orientation":6.11803674697876},{"x":52.25,"y":45.58000183105469,"team":"home_team","jersey_number":8,"v":2.5124688148498535,"orientation":6.183516502380371},{"x":76.5199966430664,"y":46.31999969482422,"team":"home_team","jersey_number":12,"v":1.457737922668457,"orientation":6.823604583740234},{"x":59.95000076293945,"y":55.47999954223633,"team":"home_team","jersey_number":13,"v":1.25,"orientation":6.2831854820251465},{"x":15.140000343322754,"y":35.2400016784668,"team":"home_team","jersey_number":30,"v":1.25,"orientation":6.2831854820251465},{"x":72.19000244140625,"y":29.56999969482422,"team":"away_team","jersey_number":28,"v":3.288236618041992,"orientation":4.86503791809082},{"x":73.95999908447266,"y":37.59000015258789,"team":"away_team","jersey_number":21,"v":4.43001127243042,"orientation":5.997133731842041},{"x":65.41000366210938,"y":26.610000610351562,"team":"away_team","jersey_number":29,"v":3.5089170932769775,"orientation":7.782674312591553},{"x":82.08999633789062,"y":41.95000076293945,"team":"away_team","jersey_number":3,"v":3.020761489868164,"orientation":5.856557846069336},{"x":66.5,"y":47.959999084472656,"team":"away_team","jersey_number":24,"v":3.25960111618042,"orientation":6.359957218170166},{"x":80.11000061035156,"y":35.310001373291016,"team":"away_team","jersey_number":13,"v":2.5124688148498535,"orientation":6.382853984832764},{"x":79.7300033569336,"y":22.049999237060547,"team":"away_team","jersey_number":15,"v":0.9013878107070923,"orientation":5.300391674041748},{"x":57.970001220703125,"y":53.34000015258789,"team":"away_team","jersey_number":19,"v":1.7677669525146484,"orientation":5.497786998748779},{"x":79.2699966430664,"y":50.689998626708984,"team":"away_team","jersey_number":17,"v":2.5124688148498535,"orientation":6.183516502380371},{"x":98.51000213623047,"y":33.58000183105469,"team":"away_team","jersey_number":1,"v":0.25,"orientation":4.71238899230957},{"x":53.72999954223633,"y":40.31999969482422,"team":"away_team","jersey_number":9,"v":3.0516388416290283,"orientation":6.893911361694336},{"x":73.02999877929688,"y":20.600000381469727,"team":"ball","jersey_number":-1,"v":13.462911605834961,"orientation":6.54617977142334}]}' output.json
# python3 pitch_control_main.py {"players":[{"x":65.66999816894531,"y":30.8799991607666,"team":"home_team","jersey_number":6,"v":1.7677669525146484,"orientation":6.1412882804870605},{"x":71.29000091552734,"y":29.510000228881836,"team":"home_team","jersey_number":19,"v":3.4003677368164062,"orientation":5.341185092926025},{"x":77.37999725341797,"y":35.66999816894531,"team":"home_team","jersey_number":18,"v":3.6912057399749756,"orientation":5.789244174957275},{"x":74.5999984741211,"y":56.84000015258789,"team":"home_team","jersey_number":11,"v":1.8027756214141846,"orientation":6.871187686920166},{"x":63.619998931884766,"y":23.25,"team":"home_team","jersey_number":23,"v":4.100305080413818,"orientation":6.938880920410156},{"x":53.47999954223633,"y":32.90999984741211,"team":"home_team","jersey_number":2,"v":1.7677669525146484,"orientation":6.425082206726074},{"x":61.95000076293945,"y":39.540000915527344,"team":"home_team","jersey_number":10,"v":3.041381359100342,"orientation":6.11803674697876},{"x":52.25,"y":45.58000183105469,"team":"home_team","jersey_number":8,"v":2.5124688148498535,"orientation":6.183516502380371},{"x":76.5199966430664,"y":46.31999969482422,"team":"home_team","jersey_number":12,"v":1.457737922668457,"orientation":6.823604583740234},{"x":59.95000076293945,"y":55.47999954223633,"team":"home_team","jersey_number":13,"v":1.25,"orientation":6.2831854820251465},{"x":15.140000343322754,"y":35.2400016784668,"team":"home_team","jersey_number":30,"v":1.25,"orientation":6.2831854820251465},{"x":72.19000244140625,"y":29.56999969482422,"team":"away_team","jersey_number":28,"v":3.288236618041992,"orientation":4.86503791809082},{"x":73.95999908447266,"y":37.59000015258789,"team":"away_team","jersey_number":21,"v":4.43001127243042,"orientation":5.997133731842041},{"x":65.41000366210938,"y":26.610000610351562,"team":"away_team","jersey_number":29,"v":3.5089170932769775,"orientation":7.782674312591553},{"x":82.08999633789062,"y":41.95000076293945,"team":"away_team","jersey_number":3,"v":3.020761489868164,"orientation":5.856557846069336},{"x":66.5,"y":47.959999084472656,"team":"away_team","jersey_number":24,"v":3.25960111618042,"orientation":6.359957218170166},{"x":80.11000061035156,"y":35.310001373291016,"team":"away_team","jersey_number":13,"v":2.5124688148498535,"orientation":6.382853984832764},{"x":79.7300033569336,"y":22.049999237060547,"team":"away_team","jersey_number":15,"v":0.9013878107070923,"orientation":5.300391674041748},{"x":57.970001220703125,"y":53.34000015258789,"team":"away_team","jersey_number":19,"v":1.7677669525146484,"orientation":5.497786998748779},{"x":79.2699966430664,"y":50.689998626708984,"team":"away_team","jersey_number":17,"v":2.5124688148498535,"orientation":6.183516502380371},{"x":98.51000213623047,"y":33.58000183105469,"team":"away_team","jersey_number":1,"v":0.25,"orientation":4.71238899230957},{"x":53.72999954223633,"y":40.31999969482422,"team":"away_team","jersey_number":9,"v":3.0516388416290283,"orientation":6.893911361694336},{"x":73.02999877929688,"y":20.600000381469727,"team":"ball","jersey_number":-1,"v":13.462911605834961,"orientation":6.54617977142334}]} output.json

