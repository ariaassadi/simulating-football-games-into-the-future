using UnityEngine;
using UnityEngine.UI;

using System;
using System.Data;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using TMPro;

using Eggs;
using Utils;

namespace GameVisualization
{

    public class GameManager : MonoBehaviour
    {
        // Game Object Spawner
        private GameObjectSpawner gameObjectSpawner;
        private GameDataLoader gameDataLoader;
        private GameLogicManager gameLogicManager;

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

        private PlayerData[] playerPositions;

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


        void FixedUpdate()
        {
            // If playing and not currently changing the game, execute Play method
            if (isPlaying && !changingGame)
            {
                this.Play();
            }
        }

        // Asynchronously load game data from the database using match ID
        public async Task<bool> LoadGameAsync(GameInfo gameInfo, int period)
        {
            // Inistialize loading screen
            Debug.Log("Loading game: " + gameInfo.MatchId + " period: " + period);
            this.gameInfo = gameInfo;

            // Remove all objects from the scene
            changingGame = true;
            DestroyObjects();
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

            gameLogicManager = new GameLogicManager(gameDataLoader, gameObjectSpawner, period);
            timeSlider.GetComponent<TimeSlider>().UpdateTimeSlider(gameLogicManager.StartFrame, gameLogicManager.EndFrame, period);
            return success;
        }

        private void DestroyObjects()
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

        // Toggle play/pause state
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

        // Set play state to false
        public void PauseGame()
        {
            isPlaying = false;
            playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = playIcon;
        }

        public void ResumeGame()
        {
            isPlaying = true;
            playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = pauseIcon;
        }


        // Fast forward the game
        public void FastForward()
        {
            PauseGame();
            gameLogicManager.MoveXFrames(25);
            UpdateUI();
        }

        // Fast backward the game
        public void FastBackward()
        {
            PauseGame();
            gameLogicManager.MoveXFrames(-25);
            UpdateUI();
        }

        // Step forward the game by one frame
        public void StepForward()
        {
            PauseGame();
            gameLogicManager.MoveXFrames(1);
            UpdateUI();
        }


        // Step backward the game by one frame
        public void StepBackward()
        {
            PauseGame();
            gameLogicManager.MoveXFrames(-1);
            UpdateUI();
        }

        // Move to a specific frame in the game
        public void MoveTo(int frameNr)
        {
            PauseGame();
            gameLogicManager.MoveTo(frameNr);
            UpdateUI();
        }

        // Update UI elements
        private void UpdateUI()
        {
            // Update the time slider
            timeSlider.GetComponent<TimeSlider>().ChangeTime(gameLogicManager.CurrentFrameNr);
        }

        public PlayerData[] GetPlayerData()
        {
            return gameLogicManager.GetPlayerData();
        }

        public bool IsGaming()
        {
            return gameObjectSpawner != null;
        }

        public GameInfo GetGameInfo()
        {
            return gameInfo;
        }
    }

}
