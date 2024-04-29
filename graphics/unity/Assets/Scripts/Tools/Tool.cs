using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using Eggs;
using Utils;
using Unity.VisualScripting;

namespace Tools
{
    public abstract class Tool : MonoBehaviour
    {
        protected GameObject[] players;
        protected Image border;
        protected ToolManager toolManager;

        private Material themeColorPrimary;

        protected bool uniqueTool = false;
        protected bool synchronized = false;

        /// <summary>
        /// Start is called before the first frame update
        /// </summary>
        protected virtual void Start()
        {
            border = transform.Find("Border").GetComponent<Image>();
            toolManager = GameObject.Find("ToolManager").GetComponent<ToolManager>();
            themeColorPrimary = Resources.Load<Material>("ThemeColorPrimaryUI");

            if (border == null)
            {
                Debug.LogError("Border not found");
                return;
            }
            else
            {
                Debug.Log("Border found for " + this.name);
            }
            if (toolManager == null)
            {
                Debug.LogError("ToolManager not found");
                return;
            }
            else
            {
                Debug.Log("ToolManager found for " + this.name);
            }
        }

        /// <summary>
        /// Select the tool.
        /// </summary>
        public virtual void Select()
        {
            border.material = new Material(Shader.Find("UI/Default"));
            border.material.color = Color.white;
        }

        /// <summary>
        /// Deselect the tool.
        /// </summary>
        public virtual void Deselect()
        {
            Destroy(border.material);
            border.material = themeColorPrimary;
        }

        /// <summary>
        /// Check if the tool is synchronized (is updated along with the game).
        /// </summary>
        public bool IsSynchronized
        {
            get { return synchronized; }
        }

        /// <summary>
        /// Check if the tool is unique (cannot be active at the same time as other tools).
        /// </summary>
        public bool IsUniqueTool
        {
            get { return uniqueTool; }
        }

        /// <summary>
        /// Select a player using mouse.
        /// </summary>
        /// <returns>The player if a player is clicked, otherwise null.</returns>
        protected GameObject SelectPlayer()
        {
            if (Input.GetMouseButtonDown(0))
            {
                RaycastHit hit;
                Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                if (Physics.Raycast(ray, out hit))
                {
                    if (hit.collider.gameObject.tag == "Player" || hit.collider.gameObject.tag == "Ball")
                    {
                        return hit.collider.gameObject;
                    }
                }
            }
            return null;
        }

        /// <summary>
        /// Deselect all players.
        /// </summary>
        /// <param name="nrPlayers"></param>
        protected void ResetPlayers(int nrPlayers)
        {
            if (players != null)
            {
                for (int i = 0; i < players.Length; i++)
                {
                    if (players[i] != null)
                    {
                        DehighlightPlayer(players[i]);
                        Debug.Log("Player deselected: " + players[i].name);
                    }
                }
            }

            players = new GameObject[nrPlayers];
        }

        /// <summary>
        /// Highlights a player.
        /// </summary>
        /// <param name="player"></param>
        protected void HighlightPlayer(GameObject player)
        {
            player.transform.GetChild(0).gameObject.SetActive(true);
            if (player.tag == "Player")
            {
                ShowPlayerName(player);
            }
            Debug.Log("Player selected: " + player.name);
        }

        /// <summary>
        /// Dehighlights a player.
        /// </summary>
        /// <param name="player"></param>
        protected void DehighlightPlayer(GameObject player)
        {
            player.transform.GetChild(0).gameObject.SetActive(false);
            if (player.tag == "Player")
            {
                HidePlayerName(player);
            }
        }

        /// <summary>
        /// Shows the player name.
        /// </summary>
        /// <param name="player"></param>
        protected void ShowPlayerName(GameObject player)
        {
            player.transform.Find("Canvas").GetComponent<EggUI>().SetTextPlayerName();
        }

        /// <summary>
        /// Hides the player name.
        /// </summary>
        /// <param name="player"></param>
        protected void HidePlayerName(GameObject player)
        {
            player.transform.Find("Canvas").GetComponent<EggUI>().SetEmptyText();
        }

        /// <summary>
        /// Runs the tool.
        /// </summary>
        public abstract void UpdateTool();
    }
}