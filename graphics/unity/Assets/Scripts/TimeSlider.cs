using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;


public class TimeSlider : MonoBehaviour
{
    [SerializeField] private GameObject timeSlider;
    [SerializeField] private TMP_Text startTime;
    [SerializeField] private TMP_Text endTime;
    private GameObject gameManager;

    [SerializeField] private GameObject gameUI;

    private int startFrame;
    private int endFrame;

    private void Start()
    {
        gameManager = GameObject.Find("GameManager");
        if (gameManager == null)
        {
            Debug.LogError("createAndMovePlayers is not assigned.");
            return;
        }

        if (!gameManager.TryGetComponent<GameManager>(out var gameManagerComponent))
        {
            Debug.LogError("GameManager component not found on GameManager GameObject.");
            return;
        }

        startFrame = gameManagerComponent.StartFrame();
        endFrame = gameManagerComponent.EndFrame();

        timeSlider.GetComponent<UnityEngine.UI.Slider>().minValue = startFrame;
        timeSlider.GetComponent<UnityEngine.UI.Slider>().maxValue = endFrame - 1;
        startTime.text = FrameToTime(startFrame).ToString();
        endTime.text = FrameToTime(endFrame).ToString();
    }


    private string FrameToTime(int frame)
    {
        int ms = frame * 40;
        int minutes = ms / 60000;
        int seconds = (ms % 60000) / 1000;

        return $"{minutes:00}:{seconds:00}";
    }

    public void ChangeTime(int frame)
    {
        timeSlider.GetComponent<UnityEngine.UI.Slider>().value = frame;
        startTime.text = FrameToTime(frame).ToString();
    }

    public void OnValueChanged()
    {
        // timeSlider.GetComponent<UnityEngine.UI.Slider>().minValue = startFrame;
        // timeSlider.GetComponent<UnityEngine.UI.Slider>().maxValue = endFrame;
        int frame = (int)timeSlider.GetComponent<UnityEngine.UI.Slider>().value;
        startTime.text = FrameToTime(frame).ToString();

        gameUI.GetComponent<TimeOverlay>().Timer(frame * 40);
        gameManager.GetComponent<GameManager>().MoveTo(frame);
    }

}
