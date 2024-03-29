using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using UnityEngine.Networking;

public class ChooseGame : MonoBehaviour
{
    private Schedule[] schedule;

    private bool scheduleIsCreated = false;
    [SerializeField] private GameObject gameOptionPrefab;

    [SerializeField] private GameObject HorizontalPanelPrefab;

    [SerializeField] private GameObject content;

    private GameObject uiManager;

    private GameObject gameManager;



    private void Start()
    {
        gameManager = GameObject.Find("GameManager");
        uiManager = GameObject.Find("UIManager");
        if (gameManager == null)
        {
            Debug.LogError("GameManager is not assigned.");
            return;
        }
        if (uiManager == null)
        {
            Debug.LogError("UIManager is not assigned.");
            return;
        }
    }

    public void GetSchedule()
    {
        string pathToDB = "";

        if (schedule == null)
        {
            if (schedule == null)
            {
                if (Application.platform == RuntimePlatform.Android && !Application.isEditor)
                {
                    pathToDB = Path.Combine(Application.persistentDataPath, "2sec_demo.sqlite");
                    if (!File.Exists(pathToDB) || new FileInfo(pathToDB).Length == 0)
                        StartCoroutine(CopyDatabase(pathToDB));
                }
                else
                    pathToDB = Path.Combine(Application.streamingAssetsPath, "2sec_demo.sqlite");

                schedule = DatabaseManager.query_schedule_db(pathToDB, "SELECT * FROM schedule");
            }

        }

        if (schedule == null)
        {
            Debug.LogError("No schedule found");
            return;
        }

        GameObject horizontalPanel;
        GameObject g;

        Debug.Log(schedule.Length);
        if (!scheduleIsCreated)
        {
            for (int i = 0; i < schedule.Length; i++)
            {
                horizontalPanel = Instantiate(HorizontalPanelPrefab, content.transform, false);
                horizontalPanel.name = $"{schedule[i].HomeTeamName} vs {schedule[i].AwayTeamName}";

                for (int j = 1; j < 3; j++)
                {
                    g = Instantiate(gameOptionPrefab, horizontalPanel.transform, false);
                    AddGameInfo(schedule[i], g, j);
                }
            }
            scheduleIsCreated = true;
        }
    }

    IEnumerator CopyDatabase(string destinationPath)
    {
        string sourcePath = Path.Combine(Application.streamingAssetsPath, "2sec_demo.sqlite");

        // Use a UnityWebRequest to copy the file from StreamingAssets to persistentDataPath
        using (UnityWebRequest www = UnityWebRequest.Get(sourcePath))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                // Write the downloaded data to persistentDataPath
                File.WriteAllBytes(destinationPath, www.downloadHandler.data);
                // Now that the database has been copied, query it
            }
            else
            {
                Debug.LogError("Failed to copy database: " + www.error);
            }
        }
    }


    private void AddGameInfo(Schedule game, GameObject g, int period)
    {
        g.GetComponentInChildren<TMPro.TextMeshProUGUI>().text = $"{game.HomeTeamName} vs {game.AwayTeamName}\nPeriod: {period}";
        if (period == 1)
        {
            g.name = $"{game.HomeTeamName} vs {game.AwayTeamName}\nFirst Half";
        }
        else
        {
            g.name = $"{game.HomeTeamName} vs {game.AwayTeamName}\nSecond Half";
        }
        // g.GetComponent<Button>().onClick.AddListener(() => uiManager.GetComponent<MenuManager>().ShowLoadingScreen());
        g.GetComponent<Button>().onClick.AddListener(() => LoadGame(game, period));
    }

    private async void LoadGame(Schedule game, int period)
    {
        // Show loading screen
        uiManager.GetComponent<MenuManager>().ShowLoadingScreen();

        // Load the game asynchronously
        bool success = await LoadGameAsync(game, period);

        // Check if loading was successful
        if (success)
        {
            // Show the game
            uiManager.GetComponent<MenuManager>().ShowGame();
        }
        else
        {
            // Show an error message or handle failure
            Debug.LogError("Failed to load game");
            // Optionally, you can hide the loading screen here as well
            // uiManager.GetComponent<MenuManager>().HideLoadingScreen();
        }
    }

    private async Task<bool> LoadGameAsync(Schedule game, int period)
    {
        // Perform the loading asynchronously
        bool success = await gameManager.GetComponent<GameManager>().LoadGameAsync(game, period);

        // Return the result
        return success;
    }

}
