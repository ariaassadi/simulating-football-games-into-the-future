using System.Collections.Generic;
using UnityEngine;
using TMPro;

using Utils;

public class ControlsUI : MonoBehaviour
{
    [SerializeField] private GameObject ControlsTamplatePrefab;

    // Two controlsTampletes in a row
    [SerializeField] private GameObject ControlsTemplateWrapperPrefab;

    [SerializeField] private GameObject content;
    private Controls[] controls;

    private bool controlsAreCreated = false;
    void Start()
    {
        string json = Resources.Load<TextAsset>("Controls").text;
        controls = JsonParser.GetControlsFromJson(json);

        foreach (Controls control in controls)
        {
            Debug.Log(control);
        }

        // Instantiate the controls
    }

    public void ShowControls()
    {
        if (controls == null)
        {
            string json = Resources.Load<TextAsset>("Controls").text;
            controls = JsonParser.GetControlsFromJson(json);
        }
        else
        {
            foreach (Controls control in controls)
            {
                Debug.Log(control);
            }
        }

        if (!controlsAreCreated)
        {
            CreateControls();
            controlsAreCreated = true;
        }
    }

    private void CreateControls()
    {
        GameObject controlsWrapper;
        GameObject controlsTemplate;
        // Show the controls
        if (controls.Length % 2 == 0)
        {
            for (int i = 0; i < controls.Length; i += 2)
            {
                // Instantiate the controls
                controlsWrapper = Instantiate(ControlsTemplateWrapperPrefab, content.transform, false);
                // Instantiate 2 controls
                for (int j = i; j < i + 2; j++)
                {
                    controlsTemplate = Instantiate(ControlsTamplatePrefab, controlsWrapper.transform, false);
                    AddControlsInfo(controlsTemplate, controls[j]);
                }
            }
        }
        else
        {
            for (int i = 0; i < controls.Length - 1; i += 2)
            {
                // Instantiate the controls
                controlsWrapper = Instantiate(ControlsTemplateWrapperPrefab, content.transform, false);
                // Instantiate 2 controls
                for (int j = i; j < i + 2; j++)
                {
                    controlsTemplate = Instantiate(ControlsTamplatePrefab, controlsWrapper.transform, false);
                    AddControlsInfo(controlsTemplate, controls[j]);
                }
            }
            // Instantiate the last control
            controlsWrapper = Instantiate(ControlsTemplateWrapperPrefab, content.transform, false);
            controlsTemplate = Instantiate(ControlsTamplatePrefab, controlsWrapper.transform, false);
            AddControlsInfo(controlsTemplate, controls[controls.Length - 1]);
            controlsTemplate = Instantiate(ControlsTamplatePrefab, controlsWrapper.transform, false);
            AddEmptyControls(controlsTemplate);
        }
    }
    private void AddControlsInfo(GameObject controlsTemplate, Controls controls)
    {
        // Add the controls info
        controlsTemplate.transform.Find("Action").GetComponentInChildren<TMP_Text>().text = controls.action;
        controlsTemplate.transform.Find("Key").GetComponentInChildren<TMP_Text>().text = controls.key;
    }

    private void AddEmptyControls(GameObject controlsTemplate)
    {
        // Add the controls info
        controlsTemplate.transform.Find("Action").GetComponentInChildren<TMP_Text>().text = "";
        controlsTemplate.transform.Find("Key").GetComponentInChildren<TMP_Text>().text = "";
    }
}
