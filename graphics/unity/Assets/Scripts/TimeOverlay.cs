using System.Data;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class TimeOverlay : MonoBehaviour
{

    [SerializeField] private TMP_Text timeText;
    public void Timer(int time)
    {
        // Get the input values
        // Update the UI text
        // GameObject.Find("Time").GetComponent<Text>().text = GetTime(time);
        timeText.text = GetTime(time);
    }

    private string GetTime(int time)
    {
        int minutes = time / 60000;
        int seconds = (time % 60000) / 1000;
        // int milliseconds = (time % 1000) / 10;

        return $"{minutes:00}:{seconds:00}";
    }
}


