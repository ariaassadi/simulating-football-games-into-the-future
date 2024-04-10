using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using Eggs;

public abstract class Tool : MonoBehaviour
{
    protected GameObject[] players;
    protected Image border;
    protected ToolManager toolManager;

    public bool uniqueTool = false;

    protected virtual void Start()
    {
        border = transform.Find("Border").GetComponent<Image>();
        toolManager = GameObject.Find("ToolManager").GetComponent<ToolManager>();
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

    protected void HighlightPlayer(GameObject player)
    {
        player.transform.GetChild(0).gameObject.SetActive(true);
        if (player.tag == "Player")
        {
            ShowPlayerName(player);
        }
        Debug.Log("Player selected: " + player.name);
    }

    protected void DehighlightPlayer(GameObject player)
    {
        player.transform.GetChild(0).gameObject.SetActive(false);
        if (player.tag == "Player")
        {
            HidePlayerName(player);
        }
    }

    protected void ShowPlayerName(GameObject player)
    {
        player.transform.Find("Canvas").GetComponent<EggUI>().SetTextPlayerName();
    }

    protected void HidePlayerName(GameObject player)
    {
        player.transform.Find("Canvas").GetComponent<EggUI>().SetEmptyText();
    }

    public virtual void Select()
    {
        border.color = Color.white;
    }

    public virtual void Deselect()
    {
        border.color = Utils.HexToColor("#12326e");
    }

    public abstract void UpdateTool();
}