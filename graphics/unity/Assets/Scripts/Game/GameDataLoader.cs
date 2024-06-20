using System.Threading.Tasks;
using System.IO;
using System.Linq;
using UnityEngine;

using Utils;
using System.Collections.Generic;

namespace GameVisualization
{
    /// <summary>
    /// Loads the game data from the database asynchronously and stores it in 
    /// the GameDataLoader object (itself).
    /// </summary>
    public class GameDataLoader
    {
        private GameInfo gameInfo;
        private Game[] gameData;
        private Game[] frameData;

        // public int StartFrame { get { return gameData.Min(x => x.Frame); } }
        public int StartFrame()
        {
            return gameData[0].frame;
        }
        public int EndFrame { get { return gameData.Max(x => x.frame); } }

        /// <summary>
        /// The frame data for the game.
        /// </summary>
        public Game[] FrameData { get { return frameData; } }

        public int SecondHalfFrame { get { return gameInfo.SecondHalfFrame; } }

        /// <summary>
        /// Initializes the GameDataLoader object with the game information.
        /// </summary>
        /// <param name="gameInfo">The game information.</param>
        public GameDataLoader(GameInfo gameInfo)
        {
            this.gameInfo = gameInfo;
        }

        /// <summary>
        /// Loads the game data asynchronously, given the period.
        /// </summary>
        /// <param name="period">The period which of the game to be loaded.</param>
        /// <returns>True, if the game is loaded successfully, false otherwise.</returns>
        public async Task<bool> LoadGameAsync(int period)
        {
            // Load game data asynchronously
            string pathToDB = GetDatabasePath();
            // string query_tracking = $"SELECT player, x, y, frame, team, orientation, jersey_number, v_x, v_y, x_future, y_future FROM games WHERE period={period} AND match_id='{gameInfo.MatchId}'";
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

        public async Task<bool> LoadGameAsyncJson()
        {
            // get clip file name from gameInfo
            string clipFileName = gameInfo.Clip;

            // Load game data asynchronously
            string pathToClips = Path.Combine(Application.streamingAssetsPath, "Clips");
            string pathToClip = Path.Combine(pathToClips, clipFileName);
            Debug.Log("Loading game data from: " + pathToClip);
            string json = await Task.Run(() => File.ReadAllText(pathToClip));
            Debug.Log("Loaded json: " + json);
            gameData = await Task.Run(() => JsonParser.GetGameFromJson(json));
            if (gameData == null || gameData.Length == 0)
            {
                Debug.LogError("Failed to load game data.");
                return false;
            }
            // frameData = (from game in gameData
            //              group game by game.Frame into frameGroup
            //              select new Game
            //              {
            //                  Frame = frameGroup.Key,
            //                  ObjectsTracked = frameGroup.First().ObjectsTracked
            //              }).ToArray();
            // if (frameData == null || frameData.Length == 0)
            // {
            //     Debug.LogError("Failed to load frame data.");
            //     return false;
            // }
            Debug.Log("Loaded game data." + gameData.Length);
            foreach (Game g in gameData)
            {
                Debug.Log(g);
            }
            return true;
        }

        /// <summary>
        /// Gets the game data for the given frame.
        /// </summary>
        /// <param name="frame">The frame.</param>
        /// <returns>The Game data for a given frame.</returns>
        public Game[] GetFrameData(int frame)
        {
            Debug.Log("Getting frame data for frame: " + frame);
            Game[] players = System.Array.FindAll(gameData, x => x.frame == frame);
            Debug.Log("Found players: " + players.Length);
            return players;
        }

        /// <summary>
        /// Gets the path to the database file.
        /// </summary>
        /// <returns>The path as a string.</returns>
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
    }
}

// public class