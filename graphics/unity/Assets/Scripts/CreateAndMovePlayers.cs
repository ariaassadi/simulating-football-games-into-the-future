using UnityEngine;
using System.Data;
using Unity.VisualScripting;

public class CreateAndMovePlayers : MonoBehaviour
{
    [SerializeField] private GameObject playerHomePrefab;
    [SerializeField] private GameObject playerAwayPrefab;
    [SerializeField] private GameObject ballPrefab;

    [SerializeField] private GameObject homeTeam;
    [SerializeField] private GameObject awayTeam;
    [SerializeField] private GameObject ball;

    private TimeOverlay timeOverlay;

    public int startFrame = 10;
    public int endFrame = 20;

    private int currentFrame;
    private int currentFrameObjectsTracked;
    private int currentFrameNr;

    private Game[] players;
    private Game[] frames;

    void Start()
    {
        timeOverlay = GetComponent<TimeOverlay>();
        // Retrieve players data
        players = DatabaseManager.query_db($"SELECT player, x, y, frame, team_direction FROM BP_vs_IKS WHERE frame>={startFrame} AND frame<={endFrame} AND period=1");

        // Retrieve frames data
        frames = DatabaseManager.query_db($"SELECT frame, objects_tracked, ms_since_start FROM BP_vs_IKS WHERE frame>={startFrame} AND frame<={endFrame} AND period=1 GROUP BY frame");

        Debug.Log(players.Length);
        Debug.Log(frames.Length);

        if (frames != null && frames.Length > 0)
        {
            // Loop through frames
            for (int i = 0; i < frames[0].ObjectsTracked; i++)
            {
                SpawnObject(players[i]);
            }

            currentFrameNr = 0;
            currentFrame = 0;
            currentFrameObjectsTracked = frames[0].ObjectsTracked;
        }
        else
        {
            Debug.Log("No frames found");
        }
    }

    float timer = 0f;
    float interval = 0.20f; // 40 milliseconds in seconds

    void Update()
    {
        if (currentFrameNr < frames.Length)
        {
            timer += Time.deltaTime;

            if (timer >= interval)
            {
                timer = 0f;

                MoveObjects();
                if (timeOverlay != null)
                {
                    if (currentFrameNr < frames.Length)
                        timeOverlay.Timer(frames[currentFrameNr].MsSinceStart);
                    // Debug.Log("Time: " + frames[currentFrameNr].MsSinceStart);
                }
                else
                    Debug.Log("TimeOverlay is null");

                currentFrameNr++;
                if (currentFrameNr < frames.Length)
                    currentFrameObjectsTracked += frames[currentFrameNr].ObjectsTracked;
            }
        }
        else
        {
            Debug.Log("Frame has completed");
        }
    }

    void MoveObjects()
    {
        Vector3 position;
        Transform playerTransform;

        for (; currentFrame < currentFrameObjectsTracked; currentFrame++)
        {
            string playerName = players[currentFrame].PlayerName;
            playerTransform = GameObject.Find(playerName)?.transform;

            if (playerTransform == null)
            {
                Debug.Log("Player not found: " + playerName);
                if (players[currentFrame].TeamDirection == "right")
                {
                    position = new Vector3(105 - players[currentFrame].X, 0, 68 - players[currentFrame].Y);
                    SpawnPlayer(position, players[currentFrame], playerAwayPrefab, awayTeam);
                }
                else
                {
                    position = new Vector3(players[currentFrame].X, 0, players[currentFrame].Y);
                    SpawnPlayer(position, players[currentFrame], playerHomePrefab, homeTeam);
                }
                playerTransform = GameObject.Find(playerName)?.transform;
            }
            else
            {
                // Cache the player's transform for reuse
                position = playerTransform.position;

                if (players[currentFrame].TeamDirection == "right")
                {
                    position.x = 105 - players[currentFrame].X;
                    position.z = 68 - players[currentFrame].Y;
                }
                else if (players[currentFrame].TeamDirection == "left")
                {
                    position.x = players[currentFrame].X;
                    position.z = players[currentFrame].Y;
                }
                else
                {
                    position.x = players[currentFrame].X;
                    position.z = players[currentFrame].Y;
                    position.y = 0.1f;
                }

                // Update the player's position
            }
            playerTransform.position = position;
        }
    }


    void SpawnObject(Game player)
    {
        Vector3 position;
        if (player.TeamDirection == "right")
        {
            position = new Vector3(105 - player.X, 0, 68 - player.Y);
            SpawnPlayer(position, player, playerAwayPrefab, awayTeam);
        }
        else if (player.TeamDirection == "left")
        {
            position = new Vector3(player.X, 0, player.Y);
            SpawnPlayer(position, player, playerHomePrefab, homeTeam);
        }
        else
        {
            position = new Vector3(player.X, 0.1f, player.Y);
            SpawnBall(position, player);
        }
    }
    void SpawnPlayer(Vector3 position, Game player, GameObject playerPrefab, GameObject team)
    {

        GameObject playerObject = Instantiate(playerPrefab, position, Quaternion.identity, team.transform) as GameObject;
        playerObject.name = player.PlayerName;
    }

    void SpawnBall(Vector3 position, Game player)
    {
        GameObject ballObject = Instantiate(ballPrefab, position, Quaternion.identity, ball.transform) as GameObject;
        ballObject.name = player.PlayerName;
    }
}
