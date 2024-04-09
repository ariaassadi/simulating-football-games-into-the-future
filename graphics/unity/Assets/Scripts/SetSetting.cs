using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;


public class Settings : MonoBehaviour
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
        }
        foreach (GameObject child in transform)
        {
            if (child.name == "Value")
            {
                settingsValue = child.GetComponent<TMP_Text>();
                Debug.Log("This name: " + this.name + "Gameobject name: " + gameObject.name);
                settingsValue.text = cameraController.GetSetting(this.name).ToString();
            }
        }
    }

    public void SetSettingValue()
    {
        settingsValue.text = cameraController.GetSetting(this.name).ToString();
    }
}
