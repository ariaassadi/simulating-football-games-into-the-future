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


namespace GameVisualization
{
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
        private int period;
        public int Period { get { return period; } }

        private int currentFrameNr;
        public int CurrentFrameNr { get { return currentFrameNr; } }

        private PlayerData[] playerData;
        private GameDataLoader gameDataLoader;
        private GameObjectSpawner gameObjectSpawner;

        private ToolManager toolManager;

        public GameLogicManager(GameDataLoader gameDataLoader, GameObjectSpawner gameObjectSpawner, int period)
        {
            this.gameDataLoader = gameDataLoader;
            this.gameObjectSpawner = gameObjectSpawner;
            this.period = period;
            startFrame = gameDataLoader.FrameData[0].Frame;
            endFrame = gameDataLoader.FrameData[gameDataLoader.FrameData.Length - 1].Frame;
            currentFrameNr = startFrame;
            SpawnFirstFrame();
        }

        private void SpawnFirstFrame()
        {
            if (gameObjectSpawner == null)
            {
                Debug.LogError("GameObjectSpawner is not assigned.");
                return;
            }
            playerData = AddPlayerData(gameDataLoader.GetFrameData(currentFrameNr));
            activePlayers = GetPlayersInFrame();
            foreach (PlayerData player in playerData)
            {
#if DEBUG_MODE
                Debug.Log(player.player_name + " " + player.x + " " + player.y);
#endif
                gameObjectSpawner.SpawnObject(player);
            }
        }

        private int VerifyFrame(int frameNr)
        {
            if (frameNr < startFrame)
                return startFrame;
            else if (frameNr > endFrame)
                return endFrame;
            else
                return frameNr;
        }


        public void MoveXFrames(int frames)
        {
            MoveTo(VerifyFrame(currentFrameNr + frames));
        }

        public void MoveTo(int frameNr)
        {
            // SetPlayFalse();
            currentFrameNr = VerifyFrame(frameNr);
            MoveObjects();
        }

        public void MoveObjects()
        {
#if DEBUG_MODE
            Debug.Log("Current frame: " + currentFrameNr);
#endif
            playerData = AddPlayerData(gameDataLoader.GetFrameData(currentFrameNr));
            string[] playersInFrame = GetPlayersInFrame();

            UpdatePlayerPositions(playersInFrame);
            HidePlayersNotInFrame(playersInFrame);
            UpdateTools();
        }

        private void UpdateTools()
        {
            // Update syncgronized tools
            AssignToolManager();
            toolManager.UpdateSynchronized();
        }

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

        private string[] GetPlayersInFrame()
        {
            return playerData.Select(player => player.player_name).ToArray();
        }

        private void UpdatePlayerPositions(string[] playersInFrame)
        {
            for (int i = 0; i < playerData.Length; i++)
            {
                UpdatePlayerPositions(playersInFrame, i);
            }
        }

        private void UpdatePlayerPositions(string[] playersInFrame, int i)
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

        private PlayerData AddPlayerData(Game game)
        {
            PlayerData player = new PlayerData();
            player.x = game.X;
            player.y = game.Y;
            player.x_future = game.X_Future;
            player.y_future = game.Y_Future;
            player.team = game.Team;
            player.offside = game.Offside;
            player.jersey_number = game.JerseyNumber;
            player.player_name = game.Player;
            // Convert x velocity and y velocity to the combined velocity
            player.v = Mathf.Sqrt(Mathf.Pow(game.V_X, 2) + Mathf.Pow(game.V_Y, 2));
            player.orientation = Mathf.Deg2Rad * (game.Orientation);
            return player;
        }

        private PlayerData[] AddPlayerData(Game[] games)
        {
            PlayerData[] players = new PlayerData[games.Length];
            for (int i = 0; i < games.Length; i++)
            {
                players[i] = AddPlayerData(games[i]);
            }
            return players;
        }

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

        public bool IsGameRunning()
        {
            return currentFrameNr < endFrame;
        }

        public PlayerData[] GetPlayerData()
        {
            return playerData;
        }
    }
}