using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using Eggs;

public class GameTools : MonoBehaviour
{
    private GameObject[] players;

    [SerializeField] private GameObject gameToolsUI;

    private GameObject selectedToolBorder;

    private Component uiManager;

    private float distance;

    private bool distanceSelected;
    private bool playerNamesSelected;
    private bool movePlayerSelected;
    private bool moveClone = false;

    private bool showPitchControl = false;

    private bool isDragging;

    private string toolSelected = "";

    private void Start()
    {
        GameObject menumanager = GameObject.Find("UIManager");
        distanceSelected = false;
        playerNamesSelected = false;
        movePlayerSelected = false;

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
        if (toolSelected == "")
        {
            SelectPlayer();
        }
        else if (toolSelected == "distance")
        {
            UpdateDistance();
        }
        else if (toolSelected == "movePlayer") // && Input.GetMouseButton(0)
        {
            UpdateMovePlayer();
        }
        else if (toolSelected == "playerNames")
        {
            // Do nothing
        }
        else
        {
            Debug.Log("No tool selected");
        }
        if (showPitchControl)
        {
            gameObject.GetComponent<PitchControl>().UpdatePitchControlTexture();
        }
    }

    public void DeselectAllTools()
    {
        distanceSelected = false;
        playerNamesSelected = false;
        toolSelected = "";
        gameToolsUI.GetComponent<GameToolsUI>().EmptyDistance();
        DeselectPlayerNames();
        DeselectMovePlayer();
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

        players = new GameObject[nrPlayers];
        distance = 0;
    }

    private void HighlightPlayer(GameObject player)
    {
        player.transform.GetChild(0).gameObject.SetActive(true);
        if (player.tag == "Player")
        {
            ShowPlayerName(player);
        }
        Debug.Log("Player selected: " + player.name);
    }

    private void DehighlightPlayer(GameObject player)
    {
        player.transform.GetChild(0).gameObject.SetActive(false);
        if (player.tag == "Player")
        {
            HidePlayerName(player);
        }
    }

    private void SelectPlayer()
    {
        if (Input.GetMouseButtonDown(0))
        {
            RaycastHit hit;
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out hit))
            {
                if (hit.collider.gameObject.tag == "Player" || hit.collider.gameObject.tag == "Ball")
                {
                    ChoosePlayer(hit.collider.gameObject);
                }
            }
        }
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
        else if (toolSelected == "movePlayer")
        {
            ChoosePlayerMove(player);
        }
        else if (toolSelected == "playerNames")
        {
            Debug.Log("Player names selected");
        }
        else
        {
            Debug.Log("No tool selected");
        }
    }

    private GameObject[] GetAllBorders()
    {
        return GameObject.FindGameObjectsWithTag("ToolBorder");
    }

    //////////////////////////
    /// DISTANCE TOOL
    /////////////////////////

    /// Updates the distance between the two selected players
    private void UpdateDistance()
    {
        if (players != null && players.Length == 2)
        {
            SelectPlayer();
            CalculateDistance();
            ShowDistance();
            PrintLineBetweenPlayers();
        }
    }


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
        gameToolsUI.GetComponent<GameToolsUI>().EmptyDistance();
    }

    private void CalculateDistance()
    {
        if (players != null && players.Length >= 2 && players[0] != null && players[1] != null)
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


    //////////////////////////
    /// END SHOW PLAYER NAME/NUMBER
    //////////////////////////

    //////////////////////////
    /// MOVE PLAYER
    //////////////////////////


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

    public void SelectMovePlayer(GameObject border)
    {
        if (!movePlayerSelected)
        {
            DeselectAllTools();
            selectedToolBorder = border;
            movePlayerSelected = true;
            toolSelected = "movePlayer";
            border.GetComponent<Image>().color = Color.white;
            players = new GameObject[2];
        }
        else
        {
            DeselectAllTools();
        }
    }

    private void DeselectMovePlayer()
    {
        movePlayerSelected = false;
        isDragging = false;
        if (selectedToolBorder != null)
        {
            selectedToolBorder.GetComponent<Image>().color = Utils.HexToColor("#12326e");
            selectedToolBorder = null;
        }
        if (players != null && players[0] != null)
        {
            SetPlayerDrag(isDragging);
        }
        RemoveClone();
    }

    private void ChoosePlayerMove(GameObject player)
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

    //////////////////////////
    /// END MOVE PLAYER
    //////////////////////////

    //////////////////////////
    /// PITCH CONTROL
    //////////////////////////

    public void SelectPitchControl(GameObject border)
    {
        if (!showPitchControl)
        {
            showPitchControl = true;
            border.GetComponent<Image>().color = Color.white;
            gameObject.GetComponent<PitchControl>().AddPlaneAndTexture();
        }
        else
        {
            DeselectPitchControl(border);
            gameObject.GetComponent<PitchControl>().RemovePlaneAndTexture();
        }
    }

    private void DeselectPitchControl(GameObject border)
    {
        showPitchControl = false;
        if (border != null)
        {
            border.GetComponent<Image>().color = Utils.HexToColor("#12326e");
        }
    }

    //////////////////////////
    /// END PITCH CONTROL
    //////////////////////////

}
