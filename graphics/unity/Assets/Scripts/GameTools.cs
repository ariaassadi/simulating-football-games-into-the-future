using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class GameTools : MonoBehaviour
{
    private GameObject[] players;

    [SerializeField] private GameObject gameToolsUI;

    private GameObject selectedToolBorder;

    private Component uiManager;

    private float distance;

    private bool distanceSelected;

    private bool playerNamesSelected;

    private string toolSelected = "";

    private void Start()
    {
        GameObject menumanager = GameObject.Find("UIManager");
        distanceSelected = false;
        playerNamesSelected = false;

        if (menumanager != null)
        {
            uiManager = menumanager.GetComponent<MenuManager>();
            if (uiManager == null)
            {
                Debug.LogError("UIManager not found");
            }
        }
        else
        {
            Debug.LogError("MenuManager not found");
        }
    }

    private void Update()
    {
        // Select the players
        if (Input.GetMouseButtonDown(0))
        {
            Debug.Log("Mouse clicked");
            RaycastHit hit;
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out hit))
            {
                if (hit.collider.gameObject.tag == "Player" || hit.collider.gameObject.tag == "Ball")
                {
                    ChoosePlayer(hit.collider.gameObject);
                }
                else
                    Debug.Log("Not a player");
            }
        }
        if (toolSelected == "distance")
        {
            if (players != null)
            {
                if (players[0] != null && players[1] != null)
                {
                    CalculateDistance();
                    ShowDistance();
                    PrintLineBetweenPlayers();
                }
            }
        }
        // if (toolSelected == "playerNames")
        // {
        //     return;
        // }
    }

    public void DeselectAllTools()
    {
        distanceSelected = false;
        playerNamesSelected = false;
        toolSelected = "";
        gameToolsUI.GetComponent<GameToolsUI>().EmptyDistance();
        DeselectPlayerNames();
        DeselectDistance();
        GameObject[] borders = GetAllBorders();
        foreach (GameObject border in borders)
        {
            border.GetComponent<Image>().color = Utils.HexToColor("#12326e");
        }
    }

    private void ResetPlayers()
    {
        players = null;
        distance = 0;
    }

    private void ResetPlayers(int nrPlayers)
    {
        for (int i = 0; i < nrPlayers; i++)
        {
            DehighlightPlayer(players[i]);
        }
        players = new GameObject[nrPlayers];
        distance = 0;
    }

    private void HighlightPlayer(GameObject player)
    {
        player.transform.GetChild(0).gameObject.SetActive(true);
        ShowPlayerName(player);
        Debug.Log("Player selected: " + player.name);
    }

    private void DehighlightPlayer(GameObject player)
    {
        player.transform.GetChild(0).gameObject.SetActive(false);
        HidePlayerName(player);
    }

    public void ChoosePlayer(GameObject player)
    {
        // Append the player to the list of players

        // Default option = no tool selected
        if (toolSelected == "")
        {
            if (players != null)
            {
                for (int i = 0; i < players.Length; i++)
                {
                    if (players[i] != null)
                    {
                        DehighlightPlayer(players[i]);
                    }
                }
            }
            players = new GameObject[1];
            HighlightPlayer(player);
            players[0] = player;
        }
        else if (toolSelected == "distance")
        {
            ChoosePlayerDistance(player);
        }
    }

    private GameObject[] GetAllBorders()
    {
        return GameObject.FindGameObjectsWithTag("ToolBorder");
    }

    //////////////////////////
    /// DISTANCE TOOL
    /////////////////////////

    public void SelectDistance(GameObject border)
    {
        if (!distanceSelected)
        {
            DeselectAllTools();
            selectedToolBorder = border;
            distanceSelected = true;
            toolSelected = "distance";
            border.GetComponent<Image>().color = Color.white;
            players = new GameObject[2];
        }
        else
        {
            DeselectAllTools();
        }
    }

    private void DeselectDistance()
    {
        distanceSelected = false;
        LineRenderer lineRenderer = GetComponent<LineRenderer>();
        if (lineRenderer != null)
        {
            Destroy(lineRenderer);
        }
        if (selectedToolBorder != null)
        {
            selectedToolBorder.GetComponent<Image>().color = Utils.HexToColor("#12326e");
            selectedToolBorder = null;
        }
        gameToolsUI.GetComponent<GameToolsUI>().EmptyDistance();
    }

    private void CalculateDistance()
    {
        distance = Vector3.Distance(players[0].transform.position, players[1].transform.position);
    }

    private void ShowDistance()
    {
        gameToolsUI.GetComponent<GameToolsUI>().ShowDistance(distance);
    }

    private void PrintLineBetweenPlayers()
    {
        if (players != null && players.Length >= 2 && players[0] != null && players[1] != null)
        {
            LineRenderer lineRenderer = GetComponent<LineRenderer>();
            if (lineRenderer == null)
            {
                lineRenderer = gameObject.AddComponent<LineRenderer>();
            }

            lineRenderer.startWidth = 0.2f;
            lineRenderer.endWidth = 0.2f;
            lineRenderer.positionCount = 2;

            Color color = Utils.HexToColor("#12326e");
            Debug.Log("Color: " + color);
            lineRenderer.material = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
            lineRenderer.material.color = color;

            lineRenderer.SetPosition(0, players[0].transform.position);
            lineRenderer.SetPosition(1, players[1].transform.position);
        }
    }

    private void RemoveLine()
    {
        LineRenderer lineRenderer = GetComponent<LineRenderer>();
        if (lineRenderer != null)
        {
            // Reset the LineRenderer
            lineRenderer.positionCount = 0;
        }
    }

    private void ChoosePlayerDistance(GameObject player)
    {
        if (players[0] == null)
        {
            HighlightPlayer(player);
            players[0] = player;
        }
        else if (players[1] == null)
        {
            HighlightPlayer(player);
            players[1] = player;
        }
        else
        {
            ResetPlayers(2);
            HighlightPlayer(player);
            players[0] = player;
            RemoveLine();
        }
    }

    //////////////////////////
    /// END DISTANCE TOOL
    //////////////////////////

    //////////////////////////
    /// SHOW PLAYER NAME/NUMBER
    //////////////////////////

    public void SelectPlayerNames(GameObject border)
    {
        if (!playerNamesSelected)
        {
            DeselectAllTools();
            selectedToolBorder = border;
            playerNamesSelected = true;
            toolSelected = "playerNames";
            border.GetComponent<Image>().color = Color.white;
            players = GameObject.FindGameObjectsWithTag("Player");
            ShowPlayerNames();
        }
        else
            DeselectAllTools();
    }

    private void DeselectPlayerNames()
    {
        if (players != null)
        {
            HidePlayerNames();
        }
        playerNamesSelected = false;
        toolSelected = "";
        if (selectedToolBorder != null)
        {
            selectedToolBorder.GetComponent<Image>().color = Utils.HexToColor("#12326e");
            selectedToolBorder = null;
        }
    }

    private void ShowPlayerNames()
    {
        if (players != null)
        {
            for (int i = 0; i < players.Length; i++)
            {
                print("Player: " + players[i].name);
                ShowPlayerName(players[i]);
                // ShowJerseyNumber(players[i]);
            }
        }
    }

    private void HidePlayerNames()
    {
        if (players != null)
        {
            foreach (GameObject player in players)
            {
                if (player != null)
                {
                    if (!isHighlighted(player))
                    {
                        HidePlayerName(player);
                    }
                }
            }
            {
            }
        }
    }

    private bool isHighlighted(GameObject player)
    {
        return player.transform.GetChild(0).gameObject.activeSelf;
    }

    private void ShowPlayerName(GameObject player)
    {
        player.transform.Find("Canvas").GetComponent<EggUI>().SetTextPlayerName();
    }

    private void HidePlayerName(GameObject player)
    {
        player.transform.Find("Canvas").GetComponent<EggUI>().SetEmptyText();
    }

    private void ShowJerseyNumber(GameObject player)
    {
        player.transform.Find("Canvas").GetComponent<EggUI>().SetTextJerseyNumber();
    }

}
