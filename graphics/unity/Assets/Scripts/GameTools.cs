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
    private string toolSelected = "";

    private void Start()
    {
        GameObject menumanager = GameObject.Find("UIManager");
        distanceSelected = false;

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

    public void DeselectAllTools()
    {
        distanceSelected = false;
        toolSelected = "";
        gameToolsUI.GetComponent<GameToolsUI>().EmptyDistance();
        DistanceDeselect();
    }

    private void CalculateDistance()
    {
        distance = Vector3.Distance(players[0].transform.position, players[1].transform.position);
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
            players[i].transform.GetChild(0).gameObject.SetActive(false);
        }
        players = new GameObject[nrPlayers];
        distance = 0;
    }


    private void ShowDistance()
    {
        gameToolsUI.GetComponent<GameToolsUI>().ShowDistance(distance);
    }

    public void DistanceSelected(GameObject border)
    {
        if (distanceSelected)
        {
            distanceSelected = false;
            ResetPlayers(2);
            players = null;
            LineRenderer lineRenderer = GetComponent<LineRenderer>();
            if (lineRenderer != null)
            {
                Destroy(lineRenderer);
            }
            border.GetComponent<Image>().color = HexToRGB("#12326e");
            toolSelected = "";
            gameToolsUI.GetComponent<GameToolsUI>().EmptyDistance();
        }
        else
        {
            selectedToolBorder = border;
            distanceSelected = true;
            toolSelected = "distance";
            border.GetComponent<Image>().color = Color.white;
            players = new GameObject[2];
        }
    }
    private void DistanceDeselect()
    {
        distanceSelected = false;
        toolSelected = "";
        if (selectedToolBorder != null)
        {
            selectedToolBorder.GetComponent<Image>().color = HexToRGB("#12326e");
            selectedToolBorder = null;
        }
        gameToolsUI.GetComponent<GameToolsUI>().EmptyDistance();
    }

    private void Update()
    {
        if (toolSelected == "distance")
        {
            // Select the players
            if (Input.GetMouseButtonDown(0))
            {
                Debug.Log("Mouse clicked");
                RaycastHit hit;
                Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                if (Physics.Raycast(ray, out hit))
                {
                    if (hit.collider.gameObject.tag == "Player")
                    {
                        ChoosePlayer(hit.collider.gameObject);
                    }
                }
            }
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
    }

    private static Color HexToRGB(string hex)
    {
        hex = hex.TrimStart('#');
        if (hex.Length != 6)
        {
            Debug.LogError("Invalid hexadecimal color code.");
            return Color.white; // Default to white color
        }

        byte r = byte.Parse(hex.Substring(0, 2), System.Globalization.NumberStyles.HexNumber);
        byte g = byte.Parse(hex.Substring(2, 2), System.Globalization.NumberStyles.HexNumber);
        byte b = byte.Parse(hex.Substring(4, 2), System.Globalization.NumberStyles.HexNumber);

        return new Color32(r, g, b, 255);
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

            Color color = HexToRGB("#12326e");
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

    public void ChoosePlayer(GameObject player)
    {
        // Append the player to the list of players
        if (players[0] == null)
        {
            player.transform.GetChild(0).gameObject.SetActive(true);
            players[0] = player;
            Debug.Log("Player 1: " + player.name);
        }
        else if (players[1] == null)
        {
            player.transform.GetChild(0).gameObject.SetActive(true);
            players[1] = player;
            Debug.Log("Player 2: " + player.name);
        }
        else
        {
            ResetPlayers(2);
            players[0] = player;
            player.transform.GetChild(0).gameObject.SetActive(true);
            Debug.Log("Player 1: " + player.name);
            RemoveLine();
        }
    }

}
