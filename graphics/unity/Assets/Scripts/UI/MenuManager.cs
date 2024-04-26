using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

using GameVisualization;

public class MenuManager : MonoBehaviour
{

    [SerializeField] private GameObject menuUI;

    [SerializeField] private GameObject gameUI;

    [SerializeField] private GameObject controlsUI;

    [SerializeField] private GameObject settingsUI;

    [SerializeField] private GameObject chooseGameUI;

    [SerializeField] private GameObject loadingScreen;

    [SerializeField] private GameObject logo;

    private GameManager gameManager;

    private CameraController cameraController;

    // "Show Game" option in menu
    private Transform showGame;

    private bool gameIsOn = false;

    private void Start()
    {
        gameManager = GameObject.Find("GameManager").GetComponent<GameManager>();
        menuUI.SetActive(true);
        logo.SetActive(true);
        showGame = menuUI.transform.Find("Show Game");
        showGame.gameObject.SetActive(false);
        gameUI.SetActive(false);
        settingsUI.SetActive(false);
        controlsUI.SetActive(false);
        chooseGameUI.SetActive(false);
        loadingScreen.SetActive(false);
    }
    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            OnEscape();
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
        logo.SetActive(false);

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
        controlsUI.GetComponent<ControlsUI>().ShowControls();
    }

    public void ShowSettings()
    {
        menuUI.SetActive(false);
        controlsUI.SetActive(false);
        gameUI.SetActive(false);
        chooseGameUI.SetActive(false);
        loadingScreen.SetActive(false);
        if (cameraController == null)
            cameraController = GameObject.Find("MainCamera").GetComponent<CameraController>();

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
        chooseGameUI.GetComponent<ChooseGame>().GetGameInfo();
    }

    public void ShowLoadingScreen()
    {
        Debug.Log("Loading Screen");
        gameUI.SetActive(true);
        loadingScreen.SetActive(true);
        menuUI.SetActive(false);
        controlsUI.SetActive(false);
        settingsUI.SetActive(false);
        chooseGameUI.SetActive(false);
    }

    private void OnEscape()
    {
        gameIsOn = gameManager.IsGaming();
        gameManager.PauseGame();
        Debug.Log("Escape");
        Debug.Log("Game is on: " + gameIsOn);
        // set all UI to false
        if (loadingScreen.activeSelf)
        {
            return;
        }

        // if game is on and on menu, show and pause game 
        if (gameIsOn && menuUI.activeSelf)
        {
            Debug.Log("Show Game");
            ShowGame();
            return;
        }

        if (settingsUI.activeSelf)
        {
            Debug.Log("Saving settings");
            cameraController.SaveSettings();
        }
        // set menuUI to true
        menuUI.SetActive(true);
        logo.SetActive(true);
        if (gameIsOn)
            showGame.gameObject.SetActive(true);
        gameUI.SetActive(false);
        settingsUI.SetActive(false);
        chooseGameUI.SetActive(false);
        controlsUI.SetActive(false);
    }

    public bool ShowingLoadingScreen()
    {
        return loadingScreen.activeSelf;
    }

    public void QuitApplication()
    {
        Application.Quit();
    }

    private void OnApplicationQuit()
    {
        // Clean up resources
        Debug.Log("Application ending after " + Time.time + " seconds");
        if (!PythonScript.CloseConnectionToPitchControlScript())
        {
            Debug.Log("Failed to close connection to pitch control script");
        }
        if (!PythonScript.StopPitchControlScript())
        {
            Debug.Log("Failed to stop pitch control script");
        }
    }


}
