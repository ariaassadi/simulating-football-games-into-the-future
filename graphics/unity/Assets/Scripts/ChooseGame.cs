using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;

public class ChooseGame : MonoBehaviour
{
    private Schedule[] schedule;
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

        if (Application.isEditor)
        {
            string pathToDB = "/home/oskarrick/uni/exjobb/simulating-football-games-into-the-future/graphics/data_processing/data/2sec.sqlite";
            schedule = DatabaseManager.query_schedule_db(pathToDB, $"SELECT * FROM schedule");
        }
        else
        {
            string pathToDB = Application.streamingAssetsPath + "/2sec_demo.sqlite";
            schedule = DatabaseManager.query_schedule_db(pathToDB, $"SELECT * FROM schedule");
        }

        if (schedule == null)
        {
            Debug.LogError("No schedule found");
            return;
        }

        GameObject horizontalPanel;
        GameObject g;

        Debug.Log(schedule.Length);
        // if (schedule.Length % 2 == 0)
        // {
        for (int i = 0; i < schedule.Length; i++)
        {
            // public static Object Instantiate(Object original, Vector3 position, Quaternion rotation, Transform parent); 

            horizontalPanel = Instantiate(HorizontalPanelPrefab, content.transform, false);
            horizontalPanel.name = $"{schedule[i].HomeTeamName} vs {schedule[i].AwayTeamName}";

            for (int j = 1; j < 3; j++)
            {
                g = Instantiate(gameOptionPrefab, horizontalPanel.transform, false);
                AddGameInfo(schedule[i], g, j);
            }
        }
        // }
        // else
        // {
        //     int lastGame = schedule.Length - 1;
        //     for (int i = 0; i < lastGame; i++)
        //     {
        //         horizontalPanel = Instantiate(HorizontalPanelPrefab, content.transform, false);

        //         for (int j = i; j < i + 2; j++)
        //         {
        //             g = Instantiate(gameOptionPrefab, horizontalPanel.transform, false);
        //             AddGameInfo(schedule[j], g, i - j + 1);
        //         }
        //     }
        //     horizontalPanel = Instantiate(HorizontalPanelPrefab, content.transform, false);
        //     g = Instantiate(gameOptionPrefab, horizontalPanel.transform, false);
        //     AddGameInfo(schedule[lastGame], g);
        // }
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
