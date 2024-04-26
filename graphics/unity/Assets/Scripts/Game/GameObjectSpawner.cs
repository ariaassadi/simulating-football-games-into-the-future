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

    public class GameObjectSpawner : MonoBehaviour
    {
        // Prefabs for players and ball
        private GameObject playerHomePrefab;
        private GameObject playerAwayPrefab;
        private GameObject ballPrefab;

        // Where the players and ball will be spawned
        private GameObject homeTeam;
        private GameObject awayTeam;
        private GameObject ball;

        public void Initialize(string homeColor, string awayColor)
        {
            homeTeam = GameObject.Find("Home Team");
            awayTeam = GameObject.Find("Away Team");
            ball = GameObject.Find("Ball");
            LoadPrefabs();
            SetTeamColor(homeColor, awayColor);
        }

        public void RemovePlayers()
        {
            // Remove all child objects of home team, away team, and ball
            DestroyChildren(homeTeam);
            DestroyChildren(awayTeam);
            DestroyChildren(ball);
        }

        private void DestroyChildren(GameObject parent)
        {
            // Destroy all child objects of a parent object
            foreach (Transform child in parent.transform)
            {
                Destroy(child.gameObject);
            }
        }

        private void LoadPrefabs()
        {
            playerHomePrefab = Resources.Load("EggPrefabHome") as GameObject;
            playerAwayPrefab = Resources.Load("EggPrefabAway") as GameObject;
            ballPrefab = Resources.Load("BallPrefab") as GameObject;
        }

        private void SetTeamColor(string homeColor, string awayColor)
        {
            if (playerHomePrefab == null || playerAwayPrefab == null)
            {
                Debug.LogError("Failed to load player prefabs.");
                return;
            }

            playerHomePrefab.GetComponent<Renderer>().sharedMaterial.color = ColorHelper.HexToColor(homeColor, 1);
            playerAwayPrefab.GetComponent<Renderer>().sharedMaterial.color = ColorHelper.HexToColor(awayColor, 1);
        }

        public void SpawnObjects(PlayerData[] playerData)
        {
            foreach (var player in playerData)
            {
                if (player.team == "home_team")
                {
                    SpawnPlayer(player, playerHomePrefab, homeTeam);
                }
                else if (player.team == "away_team")
                {
                    SpawnPlayer(player, playerAwayPrefab, awayTeam);
                }
                else
                {
                    SpawnBall(player, ballPrefab);
                }
            }
        }

        public void SpawnObject(PlayerData playerData)
        {
            if (playerData.team == "home_team")
            {
                SpawnPlayer(playerData, playerHomePrefab, homeTeam);
            }
            else if (playerData.team == "away_team")
            {
                SpawnPlayer(playerData, playerAwayPrefab, awayTeam);
            }
            else
            {
                SpawnBall(playerData, ballPrefab);
            }
        }

        private void SpawnPlayer(PlayerData player, GameObject playerPrefab, GameObject spawn)
        {
            if (playerPrefab == null)
            {
                Debug.LogError("Failed to load player prefab.");
                return;
            }

            Vector3 position = new Vector3(player.x, 0, player.y);
            GameObject playerObject = Instantiate(playerPrefab, position, Quaternion.Euler(0, player.orientation, 0), spawn.transform) as GameObject;
            playerObject.name = player.player_name;
            playerObject.tag = "Player";
            // playerObject.GetComponent<Renderer>().sharedMaterial.color = ColorHelper.HexToColor(teamColor, brightness);
            playerObject.transform.GetChild(0).gameObject.SetActive(false);
            playerObject.transform.GetChild(1).gameObject.GetComponent<EggUI>().SetJerseyNumber(player.jersey_number.ToString());
            playerObject.transform.GetChild(1).gameObject.GetComponent<EggUI>().SetPlayerName(player.player_name);
            playerObject.transform.GetChild(1).gameObject.GetComponent<EggUI>().SetEmptyText();

        }

        private void SpawnBall(PlayerData player, GameObject ballPrefab)
        {
            if (ballPrefab == null)
            {
                Debug.LogError($"Failed to load ball prefab.");
                return;
            }

            Vector3 position = new Vector3(player.x, 0.1f, player.y);
            GameObject ballObject = Instantiate(ballPrefab, position, Quaternion.identity, ball.transform) as GameObject;
            // ballObject.GetComponent<Renderer>().material.color = new Color(0.5f, 0.5f, 0.5f);
            ballObject.name = player.player_name;
            ballObject.tag = "Ball";
            ballObject.transform.GetChild(0).gameObject.SetActive(false);
        }
    }
}