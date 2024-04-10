using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

public class CalculateDistanceTool : Tool
{
    private float distance;
    private LineRenderer lineRenderer;

    /// <summary>
    /// Overriding the Start method from the Tool class to initialize the players array.
    /// </summary>
    protected override void Start()
    {
        base.Start();
        players = new GameObject[2];
    }

    /// <summary>
    /// Overriding the Select method from the Tool class to set the players array in the ToolManager.
    /// </summary>
    public override void Select()
    {
        base.Select();
        Debug.Log("Distance tool selected");
    }

    public override void Deselect()
    {
        base.Deselect();
        ResetPlayers(2);
        RemoveLine();
        EmptyDistance();
        Debug.Log("Distance tool deselected");
    }

    public override void UpdateTool()
    {
        GameObject player = SelectPlayer();
        if (player != null)
        {
            ChoosePlayer(player);
        }
        CalculateDistance(players[0], players[1]);
        ShowDistance();
        PrintLineBetweenPlayers(players[0], players[1]);

    }

    private void ChoosePlayer(GameObject player)
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
            Debug.Log("Resetting players");
            ResetPlayers(2);
            HighlightPlayer(player);
            players[0] = player;
            RemoveLine();
        }
    }

    public float CalculateDistance(GameObject player1, GameObject player2)
    {
        if (player1 == null || player2 == null)
        {
            return 0;
        }
        distance = Vector3.Distance(player1.transform.position, player2.transform.position);

        return distance;
    }

    public void ShowDistance()
    {
        GameObject.Find("Value").GetComponent<TMP_Text>().text = $"{distance.ToString("F1")}m";
    }

    public void EmptyDistance()
    {
        GameObject.Find("Value").GetComponent<TMP_Text>().text = "";
    }

    private void PrintLineBetweenPlayers(GameObject player1, GameObject player2)
    {
        if (player1 == null || player2 == null)
        {
            return;
        }
        lineRenderer = GetComponent<LineRenderer>();
        if (lineRenderer == null)
        {
            lineRenderer = gameObject.AddComponent<LineRenderer>();
        }

        lineRenderer.startWidth = 0.2f;
        lineRenderer.endWidth = 0.2f;
        lineRenderer.positionCount = 2;

        Color color = Utils.HexToColor("#12326e");
        lineRenderer.material = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        lineRenderer.material.color = color;

        lineRenderer.SetPosition(0, player1.transform.position);
        lineRenderer.SetPosition(1, player2.transform.position);
    }

    private void RemoveLine()
    {
        if (lineRenderer != null)
        {
            // Reset the LineRenderer
            lineRenderer.positionCount = 0;
        }
    }
}