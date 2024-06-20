using UnityEngine;
using System.Collections;
using System.IO;
using UnityEngine.Networking;

using Eggs;
using Utils;
using GameVisualization;

namespace Tools
{

    /// <summary>
    /// The FuturePositionTool class is responsible for displaying the future positions of the players.
    /// </summary>
    public class FuturePositionTool : Tool
    {
        private GameManager gameManager;

        protected override void Start()
        {
            base.Start();
            synchronized = true;
            gameManager = GameObject.Find("GameManager").GetComponent<GameManager>();
        }

        /// <summary>
        /// Selects the tool and activates it.
        /// </summary>
        public override void Select()
        {
            base.Select();
            UpdateFuturePosition();
            Debug.Log("Future position tool selected");
        }

        /// <summary>
        /// Deselects the tool and deactivates it.
        /// </summary>
        public override void Deselect()
        {
            base.Deselect();
            GameObject[] clones = GameObject.FindGameObjectsWithTag("Clone");
            if (clones == null)
            {
                Debug.Log("No clones found");
                return;
            }
            foreach (GameObject clone in clones)
            {
                Destroy(clone);
            }
            Debug.Log("Future position tool deselected");
        }

        /// <summary>
        /// Runs the tool.
        /// </summary>
        public override void UpdateTool()
        {
            UpdateFuturePosition();
        }

        /// <summary>
        /// Updates the future position of the players.
        /// </summary>
        private void UpdateFuturePosition()
        {
            PlayerData[] playerData = gameManager.GetComponent<GameManager>().GetPlayerData();

            // Remove existing clones
            GameObject[] clones = GameObject.FindGameObjectsWithTag("Clone");
            if (clones != null)
            {
                foreach (GameObject clone in clones)
                {
                    Debug.Log("Destroying clone " + clone.name);
                    Destroy(clone);
                }
            }
            // Create clones at future positions
            foreach (PlayerData player in playerData)
            {
                ClonePlayer(player);
            }
            Debug.Log("Updating future position tool");
        }

        /// <summary>
        /// Clones the player object and places it at the future position, then draws a line between the player and the clone.
        /// </summary>
        /// <param name="player"></param>
        private void ClonePlayer(PlayerData player)
        {
            GameObject playerObject = GameObject.Find(player.player_name);
            Debug.Log(player.ToString());

            if (playerObject == null || playerObject.tag != "Player")
            {
                Debug.Log("Player object not found");
                return;
            }
            if (player.y_future != 0 && player.y_future != 0)
            {
                GameObject playerClone = Instantiate(playerObject, new Vector3(player.x_future, 0, player.y_future), Quaternion.identity);
                Debug.Log("Cloning player " + player.player_name + " at position " + player.x_future + ", " + player.y_future);

                playerClone.name = player.player_name + " Future";
                playerClone.tag = "Clone";
                playerClone.GetComponent<Renderer>().material.color = ColorHelper.ChangeBrightness(playerObject.GetComponent<Renderer>().material.color, 0.75f);
                playerObject.transform.GetChild(0).gameObject.SetActive(false);
                playerObject.transform.GetChild(1).gameObject.GetComponent<EggUI>().SetJerseyNumber(player.jersey_number.ToString());
                playerObject.transform.GetChild(1).gameObject.GetComponent<EggUI>().SetPlayerName(player.player_name);
                playerObject.transform.GetChild(1).gameObject.GetComponent<EggUI>().SetEmptyText();
                // Add a line between the player and the clone
                LineRenderer lineRenderer = playerClone.AddComponent<LineRenderer>();
                AddLineBetweenPlayers(playerObject, playerClone, lineRenderer);
            }
            else
            {
                Debug.Log("Future position not found for player " + player.player_name);
            }
        }

        /// <summary>
        /// Adds a line between the player and the clone.
        /// </summary>
        /// <param name="player"></param>
        /// <param name="playerClone"></param>
        /// <param name="lineRenderer"></param>
        private void AddLineBetweenPlayers(GameObject player, GameObject playerClone, LineRenderer lineRenderer)
        {
            lineRenderer.startWidth = 0.1f;
            lineRenderer.endWidth = 0.1f;
            lineRenderer.positionCount = 2;
            lineRenderer.SetPosition(0, player.transform.position);
            lineRenderer.SetPosition(1, playerClone.transform.position);
            lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
            lineRenderer.startColor = Color.black;
            lineRenderer.endColor = Color.black;
        }
    }
}