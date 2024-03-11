using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class MenuManager : MonoBehaviour
{

    [SerializeField] private GameObject menuUI;

    [SerializeField] private GameObject gameUI;

    [SerializeField] private GameObject controlsUI;

    [SerializeField] private GameObject settingsUI;

    [SerializeField] private GameObject chooseGameUI;

    [SerializeField] private GameObject loadingScreen;

    private void Start()
    {
        menuUI.SetActive(true);
        gameUI.SetActive(false);
        settingsUI.SetActive(false);
        controlsUI.SetActive(false);
        chooseGameUI.SetActive(false);
        loadingScreen.SetActive(false);
    }
    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape) && !menuUI.activeSelf)
        {
            menuUI.SetActive(true);
        }
        // gameUI.SetActive(!gameUI.activeSelf);
    }

    public void ShowGame()
    {
        menuUI.SetActive(false);
        controlsUI.SetActive(false);
        settingsUI.SetActive(false);
        chooseGameUI.SetActive(false);
        loadingScreen.SetActive(false);

        gameUI.SetActive(true);
    }

    public void ShowControls()
    {
        menuUI.SetActive(false);
        gameUI.SetActive(false);
        settingsUI.SetActive(false);
        chooseGameUI.SetActive(false);
        loadingScreen.SetActive(false);

        controlsUI.SetActive(true);

    }

    public void ShowSettings()
    {
        menuUI.SetActive(false);
        controlsUI.SetActive(false);
        gameUI.SetActive(false);
        chooseGameUI.SetActive(false);
        loadingScreen.SetActive(false);

        settingsUI.SetActive(true);
    }

    public void ShowChooseGame()
    {
        menuUI.SetActive(false);
        controlsUI.SetActive(false);
        gameUI.SetActive(false);
        settingsUI.SetActive(false);
        loadingScreen.SetActive(false);

        chooseGameUI.SetActive(true);
        chooseGameUI.GetComponent<ChooseGame>().GetSchedule();
    }

    public void ShowLoadingScreen()
    {
        Debug.Log("Loading Screen");
        loadingScreen.SetActive(true);
        menuUI.SetActive(false);
        controlsUI.SetActive(false);
        gameUI.SetActive(false);
        settingsUI.SetActive(false);
        chooseGameUI.SetActive(false);

    }

    public bool ShowingLoadingScreen()
    {
        return loadingScreen.activeSelf;
    }

    public void QuitApplication()
    {
        Application.Quit();
    }

}
