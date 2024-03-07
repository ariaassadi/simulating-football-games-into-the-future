using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;


public class TimeSlider : MonoBehaviour
{
    [SerializeField] private GameObject timeSlider;
    [SerializeField] private TMP_Text startTime;
    [SerializeField] private TMP_Text endTime;
    [SerializeField] private GameObject createAndMovePlayers;

    private int startFrame;
    private int endFrame;

    private void Start()
    {
        if (createAndMovePlayers == null)
        {
            Debug.LogError("createAndMovePlayers is not assigned.");
            return;
        }

        if (!createAndMovePlayers.TryGetComponent<CreateAndMovePlayers>(out var createAndMovePlayersComponent))
        {
            Debug.LogError("CreateAndMovePlayers component not found on createAndMovePlayers GameObject.");
            return;
        }

        startFrame = createAndMovePlayersComponent.StartFrame();
        endFrame = createAndMovePlayersComponent.EndFrame();

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

        createAndMovePlayers.GetComponent<TimeOverlay>().Timer(frame * 40);
        createAndMovePlayers.GetComponent<CreateAndMovePlayers>().MoveTo(frame);
    }

}
