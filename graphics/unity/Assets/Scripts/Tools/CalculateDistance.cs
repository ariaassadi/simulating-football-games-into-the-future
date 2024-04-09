using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

public class CalculateDistance : Tool
{
    private float distance;
    private LineRenderer lineRenderer;

    public override void Select()
    {
        // this.border = transform.Find("Border").GetComponent<Image>();
        border.color = Color.white;

        Debug.Log("Distance tool selected");
    }

    public override void Deselect()
    {
        border.color = Utils.HexToColor("#12326e");

        Debug.Log("Distance tool deselected");
    }

    public override void Update()
    {
        Debug.Log("Calculating distance");
    }

    public float Distance(GameObject player1, GameObject player2)
    {
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
        lineRenderer = GetComponent<LineRenderer>();
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