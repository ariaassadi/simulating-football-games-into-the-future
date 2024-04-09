using System.Data;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class TimeOverlay : MonoBehaviour
{

    [SerializeField] private TMP_Text timeText;
    public void Timer(int time)
    {
        timeText.text = time.ToString();
    }
}


