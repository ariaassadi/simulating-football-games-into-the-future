using UnityEngine;

using Eggs;

namespace Tools
{
    /// <summary>
    /// The MovePlayerTool class is responsible for moving the players.
    /// </summary>
    public class MovePlayerTool : Tool
    {
        private bool isDragging = false;

        private bool moveClone = false;

        protected override void Start()
        {
            base.Start();
            uniqueTool = true;
        }

        public override void Select()
        {
            base.Select();

            Debug.Log("Move player tool selected");
        }

        public override void Deselect()
        {
            base.Deselect();
            ResetPlayers(2);
            Debug.Log("Move player tool deselected");
        }

        /// <summary>
        /// Runs the tool.
        /// </summary>
        public override void UpdateTool()
        {
            UpdateMovePlayer();
        }

        private void UpdateMovePlayer()
        {
            if (Input.GetMouseButtonDown(0) && !isDragging)
            {

                isDragging = true;
                Debug.Log("Mouse clicked");
                RaycastHit hit;
                Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                if (Physics.Raycast(ray, out hit))
                {
                    if (hit.collider.gameObject.tag == "Player")
                    {
                        ChoosePlayer(hit.collider.gameObject);
                    }
                    else if (hit.collider.gameObject.tag == "Clone" && moveClone)
                    {
                        Debug.Log("Clone selected" + hit.collider.gameObject.name + "is dragging?" + isDragging);
                        SetChoosePlayerMove(hit.collider.gameObject);
                        hit.collider.gameObject.GetComponent<Egg>().SetIsDragging(isDragging);
                    }
                    else
                        Debug.Log("Not a player");

                }
            }
            else if (Input.GetMouseButtonUp(0))
            {
                isDragging = false;
                SetPlayerDrag(isDragging);
                RemoveClone();
            }
        }

        private void ChoosePlayer(GameObject player)
        {
            if (players != null)
            {
                for (int i = 0; i < players.Length; i++)
                {
                    if (players[i] != null)
                    {
                        DehighlightPlayer(players[i]);
                        RemoveClone();
                    }
                }
            }
            players = new GameObject[2];
            HighlightPlayer(player);
            players[0] = player;
            SetPlayerDrag(isDragging);

            // Store reference to the cloned player
            GameObject clonedPlayer = ClonePlayer(player, player.transform.position);
            players[1] = clonedPlayer;
        }

        private void SetChoosePlayerMove(GameObject player)
        {
            if (players.Length > 1)
            {
                players[1] = player;
            }
            else
            {
                Debug.Log("No player selected");
            }
        }

        private GameObject ClonePlayer(GameObject player, Vector3 position)
        {
            // Clear clone if it exists
            RemoveClone();

            GameObject newPlayer = Instantiate(player, position, Quaternion.identity);

            // Optionally, you can adjust the properties of the copied player here
            newPlayer.tag = "Clone";

            if (moveClone)
            {
                newPlayer.GetComponent<Egg>().SetIsDragging(isDragging);
            }
            players[1] = newPlayer;

            return newPlayer; // Return the reference to the cloned player
        }

        private void RemoveClone()
        {
            if (players != null && players.Length > 1 && players[1] != null && players[1].tag == "Clone")
            {
                Debug.Log("Destroying clone");
                Destroy(players[1]);
            }

        }

        private void SetCloneDrag(bool isDragging)
        {
            if (players != null && players.Length > 1 && players[1] != null)
            {
                players[1].GetComponent<Egg>().SetIsDragging(isDragging);
            }
        }

        private void SetPlayerDrag(bool isDragging)
        {
            if (players != null && players.Length > 0 && players[0] != null)
            {
                players[0].GetComponent<Egg>().SetIsDragging(isDragging);
            }
        }

    }
}