using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using UnityEngine.Networking;

namespace GameVisualization
{
    /// <summary>
    /// Responsible for creating the game options in the UI and loading the
    /// selected game. The game options are created based on the games
    /// </summary>
    public class ChooseGame : MonoBehaviour
    {
        private GameInfo[] games;

        private bool gamesAreCreated = false;
        [SerializeField] private GameObject gameOptionPrefab;

        [SerializeField] private GameObject HorizontalPanelPrefab;

        [SerializeField] private GameObject content;

        private GameObject uiManager;

        private GameObject gameManager;

        private void Start()
        {
            gameManager = GameObject.Find("GameManager");
            uiManager = GameObject.Find("UIManager");
            if (gameManager == null)
            {
                Debug.LogError("GameManager is not assigned.");
                return;
            }
            if (uiManager == null)
            {
                Debug.LogError("UIManager is not assigned.");
                return;
            }
        }

        /// <summary>
        /// Get the game information from the database and create the game options.
        /// </summary>
        // public void GetGameInfo()
        // {
        //     string pathToDB;

        //     if (games == null)
        //     {
        //         if (games == null)
        //         {
        //             if (Application.platform == RuntimePlatform.Android && !Application.isEditor)
        //             {
        //                 pathToDB = Path.Combine(Application.persistentDataPath, "2sec_demo.sqlite");
        //                 if (!File.Exists(pathToDB) || new FileInfo(pathToDB).Length == 0)
        //                     StartCoroutine(CopyDatabase(pathToDB));
        //             }
        //             else
        //                 pathToDB = Path.Combine(Application.streamingAssetsPath, "2sec_demo.sqlite");

        //             games = DatabaseManager.query_games_db(pathToDB, "SELECT * FROM schedule");
        //         }

        //     }

        //     if (games == null)
        //     {
        //         Debug.LogError("No schedule found");
        //         return;
        //     }

        //     GameObject horizontalPanel;
        //     GameObject g;

        //     Debug.Log(games.Length);
        //     if (!gamesAreCreated)
        //     {
        //         for (int i = 0; i < games.Length; i++)
        //         {
        //             horizontalPanel = Instantiate(HorizontalPanelPrefab, content.transform, false);
        //             horizontalPanel.name = $"{games[i].HomeTeamName} vs {games[i].AwayTeamName}";

        //             for (int j = 1; j < 3; j++)
        //             {
        //                 g = Instantiate(gameOptionPrefab, horizontalPanel.transform, false);
        //                 AddGameInfo(games[i], g, j);
        //             }
        //         }
        //         gamesAreCreated = true;
        //     }
        // }

        // Create game options from json files
        public void GetGameInfo()
        {
            string pathToDB;
            string pathToClips = Path.Combine(Application.streamingAssetsPath, "Clips");
            string[] files = Directory.GetFiles(pathToClips, "*.json");

            if (files == null)
            {
                Debug.LogError("No game files found");
                return;
            }



            if (games == null)
            {
                if (games == null)
                {
                    if (Application.platform == RuntimePlatform.Android && !Application.isEditor)
                    {
                        pathToDB = Path.Combine(Application.persistentDataPath, "2sec_demo_clips.sqlite");
                        if (!File.Exists(pathToDB) || new FileInfo(pathToDB).Length == 0)
                            StartCoroutine(CopyDatabase(pathToDB));
                    }
                    else
                        pathToDB = Path.Combine(Application.streamingAssetsPath, "2sec_demo_clips.sqlite");

                    games = DatabaseManager.query_games_db(pathToDB, "SELECT * FROM schedule");
                }

            }

            if (games == null)
            {
                Debug.LogError("No schedule found");
                return;
            }

            GameObject horizontalPanel;
            GameObject g;

            Debug.Log(games.Length);
            if (!gamesAreCreated)
            {
                for (int i = 0; i < games.Length; i++)
                {
                    horizontalPanel = Instantiate(HorizontalPanelPrefab, content.transform, false);
                    horizontalPanel.name = $"{games[i].HomeTeamName} vs {games[i].AwayTeamName}";

                    g = Instantiate(gameOptionPrefab, horizontalPanel.transform, false);
                    AddGameInfoJSON(games[i], g);
                }
                gamesAreCreated = true;
            }
        }

        /// <summary>
        /// Copy the database from the StreamingAssets folder to the persistentDataPath.
        /// </summary>
        /// <param name="destinationPath">The path to copy the database to.</param>
        /// <returns></returns>
        /// <remarks>
        /// This method is used to copy the database from the StreamingAssets folder to the
        /// persistentDataPath. This is necessary because the database in the StreamingAssets
        /// folder is not readable when using android, so we need to copy it to a writeable location in order to
        /// query it.
        /// </remarks>
        IEnumerator CopyDatabase(string destinationPath)
        {
            string sourcePath = Path.Combine(Application.streamingAssetsPath, "2sec_demo_clips.sqlite");

            // Use a UnityWebRequest to copy the file from StreamingAssets to persistentDataPath
            using (UnityWebRequest www = UnityWebRequest.Get(sourcePath))
            {
                yield return www.SendWebRequest();

                if (www.result == UnityWebRequest.Result.Success)
                {
                    // Write the downloaded data to persistentDataPath
                    File.WriteAllBytes(destinationPath, www.downloadHandler.data);
                    // Now that the database has been copied, query it
                }
                else
                {
                    Debug.LogError("Failed to copy database: " + www.error);
                }
            }
        }

        /// <summary>
        /// Add the game information to the game option.
        /// </summary>
        /// <param name="game">The game game information for a game.</param>
        /// <param name="g">The game option object where the game information is visualized.</param>
        /// <param name="period">The period of the game. </param>
        private void AddGameInfoJSON(GameInfo game, GameObject g)
        {
            string timeRange = ParseTimeRangeFromJson(game);
            g.GetComponentInChildren<TMPro.TextMeshProUGUI>().text = $"{game.HomeTeamName} vs {game.AwayTeamName}\nTime range: {timeRange}";
            g.name = $"{game.HomeTeamName} vs {game.AwayTeamName}\nFrame range: {timeRange}";

            // g.GetComponent<Button>().onClick.AddListener(() => uiManager.GetComponent<MenuManager>().ShowLoadingScreen());
            g.GetComponent<Button>().onClick.AddListener(() => LoadGame(game, 2));
        }

        private string ParseTimeRangeFromJson(GameInfo game)
        {
            string startTime = game.Clip.Split('_')[1];
            startTime = startTime.Replace('-', ':');
            string endTime = game.Clip.Split('_')[2].Replace(".json", "");
            endTime = endTime.Replace('-', ':');
            return $"{startTime} - {endTime}";
        }
        /// <summary>
        /// Loads the game asynchronously and shows a loading screen while loading the game data.
        /// </summary>
        /// <param name="game">Da gaem information.</param>
        /// <param name="period"></param>
        private async void LoadGame(GameInfo game, int period)
        {
            // Show loading screen
            uiManager.GetComponent<MenuManager>().ShowLoadingScreen();

            // Load the game asynchronously
            bool success = await LoadGameAsync(game, period);

            // Check if loading was successful
            if (success)
            {
                // Show the game
                uiManager.GetComponent<MenuManager>().ShowGame();
            }
            else
            {
                // Show an error message or handle failure
                Debug.LogError("Failed to load game");
                // Optionally, you can hide the loading screen here as well
                // uiManager.GetComponent<MenuManager>().HideLoadingScreen();
            }
        }

        /// <summary>
        /// Loads the game asynchronously calling on GameManager.
        /// </summary>
        /// <param name="game"></param>
        /// <param name="period"></param>
        /// <returns>True if successfull, false otherwise.</returns>
        private async Task<bool> LoadGameAsync(GameInfo game, int period)
        {
            // Perform the loading asynchronously
            bool success = await gameManager.GetComponent<GameManager>().LoadGameAsync(game, period);

            // Return the result
            return success;
        }

    }
}
