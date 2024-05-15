// #define DEBUG_MODE

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
using Tools;


namespace GameVisualization
{
    /// <summary>
    /// The GameManager class is responsible for managing the game state logic.
    /// </summary>
    public class GameLogicManager
    {
        // Handle missing players and subs
        private string[] activePlayers;

        private string[] inactivePlayerNames;
        private GameObject[] inactivePlayerObjects;

        private int startFrame;
        public int StartFrame { get { return startFrame; } }
        private int endFrame;
        public int EndFrame { get { return endFrame; } }

        private int currentFrameNr;
        public int CurrentFrameNr { get { return currentFrameNr; } }

        private PlayerData[] playerData;
        private GameDataLoader gameDataLoader;
        private GameObjectSpawner gameObjectSpawner;

        private ToolManager toolManager;

        /// <summary>
        /// Initializes a new instance of the <see cref="GameLogicManager"/> class.
        /// </summary>
        /// <param name="gameDataLoader">The game data loader <see cref="GameDataLoader"/>.</param>
        /// <param name="gameObjectSpawner">The game object spawner <see cref="GameObjectSpawner"/>.</param>
        public GameLogicManager(GameDataLoader gameDataLoader, GameObjectSpawner gameObjectSpawner)
        {
            this.gameDataLoader = gameDataLoader;
            this.gameObjectSpawner = gameObjectSpawner;
            startFrame = gameDataLoader.StartFrame();
            endFrame = gameDataLoader.EndFrame;
            currentFrameNr = startFrame;
            // Spawn the first frame to fill the scene.
            SpawnFirstFrame();
        }

        /// <summary>
        /// Spawns the first frame to fill the scene with players at the first frame. 
        /// </summary>
        private void SpawnFirstFrame()
        {
            if (gameObjectSpawner == null)
            {
                Debug.LogError("GameObjectSpawner is not assigned.");
                return;
            }
            playerData = AddPlayerData(gameDataLoader.GetFrameData(currentFrameNr));
            foreach (PlayerData player in playerData)
            {
                Debug.Log("Player: " + player);
            }
            activePlayers = GetPlayersInFrame();
            gameObjectSpawner.SpawnObjects(playerData);
        }

        /// <summary>
        /// Verifies that the frame number is within the range of the game data.
        /// </summary>
        /// <param name="frameNr">The frame number</param>
        /// <returns>The frameNr if it is in range, otherwise the appropriate extremities.</returns>
        private int VerifyFrame(int frameNr)
        {
            if (frameNr < startFrame)
                return startFrame;
            else if (frameNr > endFrame)
                return endFrame;
            else
                return frameNr;
        }

        /// <summary>
        /// Moves the game forward by a number of frames (backwards if the number is negative).
        /// </summary>
        /// <param name="frames">The number of frames to move.</param>
        public void MoveXFrames(int frames)
        {
            MoveTo(VerifyFrame(currentFrameNr + frames));
        }

        /// <summary>
        /// Moves the game to a specific frame.
        /// </summary>
        /// <param name="frameNr">The frame to move to.</param>
        public void MoveTo(int frameNr)
        {
            // SetPlayFalse();
            currentFrameNr = VerifyFrame(frameNr);
            MoveObjects();
        }

        /// <summary>
        /// Moves the game objects according to the current frame.
        /// </summary>
        private void MoveObjects()
        {
#if DEBUG_MODE
            Debug.Log("Current frame: " + currentFrameNr);
#endif
            playerData = AddPlayerData(gameDataLoader.GetFrameData(currentFrameNr));
            string[] playersInFrame = GetPlayersInFrame();

            // Update player positions
            UpdatePlayerPositions();
            // Hide players not in frame
            HidePlayersNotInFrame(playersInFrame);
            // Update tools as necessary
            UpdateTools();
        }

        /// <summary>
        /// Updates the positions of all players in a frame.
        /// </summary>
        private void UpdatePlayerPositions()
        {
            for (int i = 0; i < playerData.Length; i++)
            {
                UpdatePlayerPositions(i);
            }
        }

        /// <summary>
        /// Updates the position of a specific player in the frame.
        /// </summary>
        /// <param name="i">The index of the player in PlayerData[]<see cref="PlayerData"/> </param>
        private void UpdatePlayerPositions(int i)
        {
            // playersInFrame[frame - currentFrameStartIndex] = playerName;
            if (i > 23)
            {
                Debug.LogError("Index out of bounds: " + i);
                // break;
            }
            string playerName = playerData[i].player_name;

            GameObject playerObject = GameObject.Find(playerName);
            MovePlayer(playerObject, playerData[i]);
        }

        /// <summary>
        /// Moves the player to the new position.
        /// </summary>
        /// <param name="playerObject">The player object to be moved.</param>
        /// <param name="player">The data about where the player object should be moved.</param>
        private void MovePlayer(GameObject playerObject, PlayerData player)
        {
            if (playerObject == null && (inactivePlayerNames == null || (inactivePlayerNames != null && Array.IndexOf(inactivePlayerNames, player.player_name) == -1)))
            {
#if DEBUG_MODE
                Debug.Log("Player not found: " + player.player_name);
#endif
                gameObjectSpawner.SpawnObject(player);
            }
            else
            {
                if (inactivePlayerObjects != null && Array.IndexOf(inactivePlayerNames, player.player_name) != -1)
                {
                    int index = Array.IndexOf(inactivePlayerNames, player.player_name);
                    playerObject = inactivePlayerObjects[index];
                    playerObject.SetActive(true);
                    RemoveFromArray(player.player_name, ref inactivePlayerNames);
                    RemoveFromArray(playerObject, ref inactivePlayerObjects);
                }
                Transform playerTransform = playerObject.transform;
                Vector3 position = playerTransform.position;

                // Move the player to the new position
                position.x = player.x;
                position.z = player.y;
                playerTransform.rotation = Quaternion.Euler(0, player.orientation + 90, 0);
                playerTransform.position = position;
            }
        }

        /// <summary>
        /// Hides the players that are not in the frame.
        /// </summary>
        /// <param name="playersInFrame">An array of players in the frame (to not hide).</param>
        private void HidePlayersNotInFrame(string[] playersInFrame)
        {
            foreach (string player in activePlayers)
            {
                if (Array.IndexOf(playersInFrame, player) == -1)
                {
                    GameObject playerObject = GameObject.Find(player);
                    if (playerObject != null)
                    {
                        // Add playerObject to inactivePlayers array
                        AddToArray(playerObject.name, ref inactivePlayerNames);
                        AddToArray(playerObject, ref inactivePlayerObjects);
                        playerObject.SetActive(false);
                    }
                    else
                    {
#if DEBUG_MODE
                        Debug.Log("Player not found: " + player);
#endif
                    }
                }
            }
        }

        /// <summary>
        /// Updates the tools in the scene.
        /// </summary>
        private void UpdateTools()
        {
            // Update syncgronized tools
            AssignToolManager();
            toolManager.UpdateSynchronized();
        }

        /// <summary>
        /// Assigns the tool manager if it is not already assigned.
        /// </summary>
        private void AssignToolManager()
        {
            if (toolManager == null)
            {
                GameObject toolManagerObject = GameObject.Find("ToolManager");
                if (toolManagerObject == null)
                {
                    Debug.LogError("ToolManager cannot be found");
                }
                else
                {
                    toolManager = toolManagerObject.GetComponent<ToolManager>();
                }
            }
        }

        /// <summary>
        /// Gets the player in the frame.
        /// </summary>
        /// <returns>A string array of player names.</returns>
        private string[] GetPlayersInFrame()
        {
            return playerData.Select(player => player.player_name).ToArray();
        }

        /// <summary>
        /// Converts the game data to player data.
        /// </summary>
        /// <param name="game">The game data.</param>
        /// <returns>The player data. </returns>
        private PlayerData AddPlayerData(Game game)
        {
            PlayerData player = new PlayerData();
            player.x = game.x;
            player.y = game.y;
            player.x_future = game.x_future;
            player.y_future = game.y_future;
            player.team = game.team;
            // player.offside = game.Offside;
            player.jersey_number = game.jersey_number;
            player.player_name = game.player;
            // Convert x velocity and y velocity to the combined velocity
            player.v = Mathf.Sqrt(Mathf.Pow(game.v_x, 2) + Mathf.Pow(game.v_y, 2));
            player.orientation = Mathf.Deg2Rad * (game.orientation);
            return player;
        }

        /// <summary>
        /// Converts the game data to player data.
        /// </summary>
        /// <param name="games">A game array data.</param>
        /// <returns>A PlayerData array.</returns>
        private PlayerData[] AddPlayerData(Game[] games)
        {
            PlayerData[] players = new PlayerData[games.Length];
            for (int i = 0; i < games.Length; i++)
            {
                players[i] = AddPlayerData(games[i]);
            }
            return players;
        }

        /// <summary>
        /// Adds an element to the specified array.
        /// </summary>
        /// <typeparam name="T">The type of the element.</typeparam>
        /// <param name="element">The element to add.</param>
        /// <param name="array">The array to add the element to.</param>
        private void AddToArray<T>(T element, ref T[] array)
        {
            if (array == null)
            {
                array = new T[1];
                array[0] = element;
            }
            else
            {
                Array.Resize(ref array, array.Length + 1);
                array[array.Length - 1] = element;
            }
        }

        /// <summary>
        /// Removes an element from the specified array.
        /// </summary>
        /// <typeparam name="T">The type of the element.</typeparam>
        /// <param name="element">The element to remove.</param>
        /// <param name="array">The array to remove the element from.</param>
        private void RemoveFromArray<T>(T element, ref T[] array)
        {
            if (array == null)
            {
                return;
            }
            else
            {
                List<T> tempList = new List<T>(array);
                tempList.Remove(element);
                array = tempList.ToArray();
            }
        }

        /// <summary>
        /// Checks if the game is currently running.
        /// </summary>
        /// <returns>True if the game is running, false otherwise.</returns>
        public bool IsGameRunning()
        {
            return currentFrameNr < endFrame;
        }

        /// <summary>
        /// Retrieves the player data.
        /// </summary>
        /// <returns>An array of PlayerData containing player information for the current frame.</returns>
        public PlayerData[] GetPlayerData()
        {
            return playerData;
        }
    }
}