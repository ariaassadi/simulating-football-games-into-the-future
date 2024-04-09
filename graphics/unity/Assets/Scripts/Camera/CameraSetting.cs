using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;


public class CameraSettings : MonoBehaviour
{
    private TMP_Text settingsValue;

    private CameraController cameraController;

    // Start is called before the first frame update
    void Start()
    {
        cameraController = GameObject.Find("MainCamera").GetComponent<CameraController>();

        if (cameraController == null)
        {
            Debug.LogError("CameraController not found");
            return;
        }
        settingsValue = transform.Find("Value").GetComponent<TMP_Text>();
        if (settingsValue == null)
        {
            Debug.LogError("Value not found");
            return;
        }
        else
        {
            Debug.Log("Value found for " + this.name);
            SetSettingValue();
        }
        // settingsValue.text = cameraController.GetSetting(this.name).ToString();
    }

    private void SetSettingValue()
    {
        string text = cameraController.GetSetting(this.name).ToString();
        Debug.Log("Setting value for " + this.name + " is " + text);
        settingsValue.text = cameraController.GetSetting(this.name).ToString();
    }

    public void ResetSettingValue()
    {
        cameraController.ResetSetting(this.name);
        SetSettingValue();
    }

    public void IncreaseSettingValue()
    {
        cameraController.IncreaseSetting(this.name);
        SetSettingValue();
    }

    public void DecreaseSettingValue()
    {
        cameraController.DecreaseSetting(this.name);
        SetSettingValue();
    }
}
