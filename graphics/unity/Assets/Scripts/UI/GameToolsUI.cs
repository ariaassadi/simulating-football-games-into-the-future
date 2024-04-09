using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameToolsUI : MonoBehaviour
{
    private GameObject gameTools;
    private GameObject player1;
    private GameObject player2;

    [SerializeField] private GameObject distanceText;


    public void ShowDistance(float distance)
    {
        distanceText.GetComponent<TMPro.TextMeshProUGUI>().text = $"{distance.ToString("F1")}m";
    }

    public void EmptyDistance()
    {
        distanceText.GetComponent<TMPro.TextMeshProUGUI>().text = "";
    }
}