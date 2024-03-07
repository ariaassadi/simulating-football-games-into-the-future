using System.Data;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class TimeOverlay : MonoBehaviour
{

    [SerializeField] private TMP_Text timeText;
    public void Timer(int frame)
    {
        // Get the input values
        // Update the UI text
        // GameObject.Find("Time").GetComponent<Text>().text = GetTime(time);
        timeText.text = GetTime(frame);
    }

    private string GetTime(int frame)
    {
        int ms = frame * 40;
        int minutes = ms / 60000;
        int seconds = (ms % 60000) / 1000;
        // int milliseconds = (time % 1000) / 10;

        return $"{minutes:00}:{seconds:00}";
    }
}


