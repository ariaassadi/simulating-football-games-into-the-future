using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;


public class TimeSlider : MonoBehaviour
{
    [SerializeField] private GameObject timeSlider;
    [SerializeField] private TMP_Text startTime;
    [SerializeField] private TMP_Text endTime;

    [SerializeField] private TMP_Text scoreBoardTime;

    private GameObject gameManager;

    [SerializeField] private GameObject gameUI;

    private int startFrame;
    private int endFrame;
    private int period;

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

        // startFrame = gameManagerComponent.StartFrame();
        // endFrame = gameManagerComponent.EndFrame();
        // period = gameManagerComponent.Period();

        // timeSlider.GetComponent<UnityEngine.UI.Slider>().minValue = startFrame;
        // timeSlider.GetComponent<UnityEngine.UI.Slider>().maxValue = endFrame - 1;
        // startTime.text = FrameToTime(startFrame, period).ToString();
        // endTime.text = FrameToTime(endFrame, period).ToString();
    }

    public void UpdateTimeSlider(int startFrame, int endFrame, int period)
    {
        timeSlider.GetComponent<UnityEngine.UI.Slider>().minValue = startFrame;
        timeSlider.GetComponent<UnityEngine.UI.Slider>().maxValue = endFrame - 1;
        this.startFrame = startFrame;
        this.endFrame = endFrame;
        this.period = period;
        startTime.text = FrameToTime(startFrame, period).ToString();
        endTime.text = FrameToTime(endFrame, period).ToString();
        scoreBoardTime.text = FrameToTime(startFrame, period).ToString();
    }

    private string FrameToTime(int frame, int period)
    {
        int ms, minutes, seconds;
        if (period == 1)
        {
            ms = frame * 40;
            minutes = ms / 60000;
            seconds = (ms % 60000) / 1000;

        }
        else
        {
            Debug.Log((frame - startFrame).ToString());
            ms = (frame - startFrame) * 40;
            minutes = ms / 60000 + 45;
            seconds = (ms % 60000) / 1000;
        }
        return $"{minutes:00}:{seconds:00}";
    }

    // Set time at frame startframe to 45:00


    public void ChangeTime(int frame)
    {
        timeSlider.GetComponent<UnityEngine.UI.Slider>().value = frame;
        startTime.text = FrameToTime(frame, period).ToString();
        scoreBoardTime.text = FrameToTime(frame, period).ToString();
    }

    public void OnValueChanged()
    {
        // timeSlider.GetComponent<UnityEngine.UI.Slider>().minValue = startFrame;
        // timeSlider.GetComponent<UnityEngine.UI.Slider>().maxValue = endFrame;
        int frame = (int)timeSlider.GetComponent<UnityEngine.UI.Slider>().value;
        startTime.text = FrameToTime(frame, period).ToString();

        scoreBoardTime.text = FrameToTime(frame, period).ToString();
        gameManager.GetComponent<GameManager>().MoveTo(frame);
    }

}
