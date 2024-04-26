using System.Threading.Tasks;
using UnityEngine;

namespace GameVisualization
{
    public class GameDataLoader
    {
        private GameInfo gameInfo;
        private Game[] gameData;
        private Game[] frameData;

        public GameInfo GameInfo { get { return gameInfo; } }
        public Game[] GameData { get { return gameData; } }
        public Game[] FrameData { get { return frameData; } }

        public GameDataLoader(GameInfo gameInfo)
        {
            this.gameInfo = gameInfo;
        }

        public async Task<bool> LoadGameAsync(int period)
        {
            // Load game data asynchronously
            string pathToDB = GetDatabasePath();
            string query_tracking = $"SELECT player, x, y, frame, team, orientation, jersey_number, offside, v_x, v_y, x_future, y_future FROM games WHERE period={period} AND match_id='{gameInfo.MatchId}'";
            string query_frames = $"SELECT frame, objects_tracked FROM games WHERE period={period} AND match_id='{gameInfo.MatchId}' GROUP BY frame";

            gameData = await Task.Run(() => DatabaseManager.query_db(pathToDB, query_tracking));
            frameData = await Task.Run(() => DatabaseManager.query_db(pathToDB, query_frames));

            if (gameData == null || gameData.Length == 0 || frameData == null || frameData.Length == 0)
            {
                Debug.LogError("Failed to load game data.");
                return false;
            }
            return true;
        }

        private string GetDatabasePath()
        {
            if (Application.platform == RuntimePlatform.Android && !Application.isEditor)
            {
                return Application.persistentDataPath + "/2sec_demo.sqlite";
            }
            else
            {
                return Application.streamingAssetsPath + "/2sec_demo.sqlite";
            }
        }

        public Game[] GetFrameData(int frame)
        {
            return System.Array.FindAll(gameData, x => x.Frame == frame);
        }

    }

}