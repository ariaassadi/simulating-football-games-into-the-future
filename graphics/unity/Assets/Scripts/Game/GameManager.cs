using UnityEngine;
using UnityEngine.UI;

using System;
using System.Data;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using TMPro;

using Eggs;
using Tools;
using Utils;

/// <summary>
/// Handles the game
/// </summary>
/// The classes in this namespace are responsible for managing and manipulating
/// the game state the game state.
namespace GameVisualization
{

    /// <summary>
    /// The GameManager class is responsible for managing the game state and 
    /// the game objects. It is responsible for loading game data, playing the
    /// game, and updating the game state. It also manages the UI elements. It
    /// creates the GameObjectSpawner, GameLogicManager, and GameDataLoader
    /// objects.
    /// </summary>
    public class GameManager : MonoBehaviour
    {
        // Game Object Spawner
        private GameObjectSpawner gameObjectSpawner;
        private GameDataLoader gameDataLoader;
        private GameLogicManager gameLogicManager;

        /// UI elements
        // The play and pause icons
        [SerializeField] private Texture2D playIcon;
        [SerializeField] private Texture2D pauseIcon;

        // The play and pause button
        [SerializeField] private GameObject playPauseButton;
        [SerializeField] private GameObject timeSlider;

        [SerializeField] private GameObject homeTeamNameShort;
        [SerializeField] private GameObject awayTeamNameShort;

        // Scripts
        [SerializeField] private GameObject gameUI;
        private GameObject uiManager;

        private GameInfo gameInfo;

        float timer = 0f;
        float interval = 0.04f; // 40 milliseconds in seconds, the interval between frames
        private bool isPlaying = false;
        private bool changingGame = false; // True when changing the game, maybe unnecessary

        private void Start()
        {
            // Find the UIManager object in the scene
            uiManager = GameObject.Find("UIManager");
            if (uiManager == null)
            {
                Debug.LogError("UIManager is not assigned.");
                return;
            }
        }

        /// <summary>
        /// FixedUpdate is called continously at a fixed interval.
        /// </summary>
        void FixedUpdate()
        {
            // If playing and not currently changing the game, execute Play method
            if (isPlaying && !changingGame)
            {
                this.Play();
            }
        }

        /// <summary>
        /// Asynchronously load game data from the database using match ID.
        /// </summary>
        /// <param name="gameInfo">Information about the game to load.</param>
        /// <param name="period">The period of the game to load.</param>
        /// <returns>Returns true if loading is successful, false otherwise.</returns>
        public async Task<bool> LoadGameAsync(GameInfo gameInfo, int period)
        {
            // Inistialize loading screen
            Debug.Log("Loading game: " + gameInfo.MatchId + " period: " + period);
            this.gameInfo = gameInfo;

            // Remove all objects from the scene
            changingGame = true;
            ResetGameState();
            changingGame = false;

            // gameUI.GetComponent<TimeOverlay>().Timer(0);
            homeTeamNameShort.GetComponentInChildren<TMP_Text>().text = gameInfo.HomeTeamNameShort;
            homeTeamNameShort.GetComponentInChildren<Image>().color = ColorHelper.HexToColor(gameInfo.HomeTeamColor);
            awayTeamNameShort.GetComponentInChildren<TMP_Text>().text = gameInfo.AwayTeamNameShort;
            awayTeamNameShort.GetComponentInChildren<Image>().color = ColorHelper.HexToColor(gameInfo.AwayTeamColor);

            // Initialize Game Data Loader
            gameDataLoader = new GameDataLoader(gameInfo);
            bool success = await gameDataLoader.LoadGameAsync(period);
            if (!success)
                return success;

            gameObjectSpawner = gameObject.AddComponent<GameObjectSpawner>();
            gameObjectSpawner.Initialize(gameInfo.HomeTeamColor, gameInfo.AwayTeamColor);

            gameLogicManager = new GameLogicManager(gameDataLoader, gameObjectSpawner);
            timeSlider.GetComponent<TimeSlider>().UpdateTimeSlider(gameLogicManager.StartFrame, gameLogicManager.EndFrame, period);
            return success;
        }

        /// <summary>
        /// Destroys game objects and resets the game state.
        /// </summary>
        private void ResetGameState()
        {
            if (gameObjectSpawner != null)
            {
                gameObjectSpawner.RemovePlayers();
                Destroy(gameObjectSpawner);
            }
            gameLogicManager = null;

            // Unload tools
            if (GameObject.Find("ToolManager"))
            {
                GameObject.Find("ToolManager").GetComponent<ToolManager>().DeselectTools();
            }
            else
            {
                Debug.LogWarning("ToolManager not found");
            }

            // Reset the time slider
            timeSlider.GetComponent<TimeSlider>().ChangeTime(0);

            System.Diagnostics.Stopwatch stopwatch = new System.Diagnostics.Stopwatch();
            stopwatch.Start();

            GC.Collect();
            GC.WaitForPendingFinalizers();

            stopwatch.Stop();
            Debug.Log("Execution time for garbage cleanup: " + stopwatch.ElapsedMilliseconds + " ms");

        }

        /// <summary>
        /// Plays the game by moving objects according to frame data.
        /// </summary>
        private void Play()
        {
            // Play the game by moving objects according to frame data
            if (gameLogicManager.IsGameRunning())
            {
                timer += Time.deltaTime;

                if (timer >= interval)
                {
                    timer = 0f;
                    gameLogicManager.MoveXFrames(1);
                    UpdateUI();
                }
            }
            else
            {
                Debug.Log("Frame has completed");
            }
        }

        /// <summary>
        /// Toggles the play/pause state of the game.
        /// </summary>
        public void PlayPause()
        {
            Debug.Log("PlayPause");
            isPlaying = !isPlaying;
            if (isPlaying)
            {
                playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = pauseIcon;
            }
            else
            {
                playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = playIcon;
            }
        }

        /// <summary>
        /// Pauses the game.
        /// </summary>
        public void PauseGame()
        {
            isPlaying = false;
            playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = playIcon;
        }

        /// <summary>
        /// Resumes the game.
        /// </summary>
        public void ResumeGame()
        {
            isPlaying = true;
            playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = pauseIcon;
        }


        /// <summary>
        /// Fast forwards the game by 25 frames (1 second at 25 fps).
        /// </summary>
        public void FastForward()
        {
            PauseGame();
            gameLogicManager.MoveXFrames(25);
            UpdateUI();
        }

        /// <summary>
        /// Fast backwards the game by 25 frames (1 second at 25 fps).
        /// </summary>
        public void FastBackward()
        {
            PauseGame();
            gameLogicManager.MoveXFrames(-25);
            UpdateUI();
        }

        /// <summary>
        /// Step forward the game by one frame.
        /// </summary>
        public void StepForward()
        {
            PauseGame();
            gameLogicManager.MoveXFrames(1);
            UpdateUI();
        }


        /// <summary>
        /// Step backward the game by one frame.
        /// </summary>
        public void StepBackward()
        {
            PauseGame();
            gameLogicManager.MoveXFrames(-1);
            UpdateUI();
        }

        /// <summary>
        /// Move to a specific frame in the game.
        /// </summary>
        /// <param name="frameNr">The frame number to move to</param>
        public void MoveTo(int frameNr)
        {
            PauseGame();
            gameLogicManager.MoveTo(frameNr);
            UpdateUI();
        }

        /// <summary>
        /// Update UI elements.
        /// </summary>
        private void UpdateUI()
        {
            // Update the time slider
            timeSlider.GetComponent<TimeSlider>().ChangeTime(gameLogicManager.CurrentFrameNr);
        }

        /// <summary>
        /// Gets the player data. 
        /// </summary>
        /// <returns></returns>
        public PlayerData[] GetPlayerData()
        {
            return gameLogicManager.GetPlayerData();
        }

        /// <summary>
        /// Checks if the game is currently being played.
        /// </summary>
        /// <returns>True if the game is active, false otherwise.</returns>
        public bool IsGaming()
        {
            return gameObjectSpawner != null;
        }

        /// <summary>
        /// Retrieves information about the game.
        /// </summary>
        /// <returns>The GameInfo object containing information about the game.</returns>
        public GameInfo GetGameInfo()
        {
            return gameInfo;
        }
    }
}
