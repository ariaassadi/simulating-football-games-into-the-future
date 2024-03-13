using UnityEngine;
using UnityEngine.UI;
using System.Data;
using System;
using System.Threading.Tasks;
using TMPro;

public class GameManager : MonoBehaviour
{

    // Prefabs for the players and the ball
    [SerializeField] private GameObject playerHomePrefab;
    [SerializeField] private GameObject playerAwayPrefab;
    [SerializeField] private GameObject playerPrefab;
    [SerializeField] private GameObject ballPrefab;

    // The game objects for the home team, away team and the ball
    [SerializeField] private GameObject homeTeam;
    [SerializeField] private GameObject awayTeam;
    [SerializeField] private GameObject ball;

    // The play and pause icons
    [SerializeField] private Texture2D playIcon;
    [SerializeField] private Texture2D pauseIcon;

    // The play and pause button
    [SerializeField] private GameObject playPauseButton;

    [SerializeField] private GameObject timeSlider;

    [SerializeField] private GameObject homeTeamNameShort;
    [SerializeField] private GameObject awayTeamNameShort;

    // Scripts
    [SerializeField] private GameObject gameUI;
    private GameObject uiManager;

    [SerializeField] private int startFrame = 0;
    [SerializeField] private int endFrame = 67500;

    private int currentFrameStartIndex;
    private int currentFrameEndIndex;
    private int currentFrameNr;

    private Game[] playerData;
    private Game[] frameData;

    private Schedule gameInfo;

    float timer = 0f;
    float interval = 0.04f; // 40 milliseconds in seconds, the interval between frames
    private bool isPlaying = false;
    private bool changingGame = false; // True when changing the game, maybe unnecessary

    private int[][] framesStartAndEndIndex;


    private void Start()
    {
        // Find the UIManager object in the scene
        uiManager = GameObject.Find("UIManager");
        if (uiManager == null)
        {
            Debug.LogError("UIManager is not assigned.");
            return;
        }
    }


    void Update()
    {
        // If playing and not currently changing the game, execute Play method
        if (isPlaying && !changingGame)
        {
            Play();
        }
    }

    // Asynchronously load game data from the database using match ID
    public async Task<bool> LoadGameAsync(string match_id)
    {
        // SQL query to retrieve player data for a specific match and period
        string query_tracking = $"SELECT player, x, y, frame, team, orientation, jersey_number FROM games WHERE frame>={startFrame} AND frame<{endFrame} AND period=1 AND match_id='{match_id}'";

        string query_frames = $"SELECT frame, objects_tracked FROM games WHERE frame>={startFrame} AND frame<{endFrame} AND period=1 AND match_id='{match_id}' GROUP BY frame";


        Debug.Log($"Select: {query_tracking} END");

        // Retrieve players data asynchronously
        Task<Game[]> playerDataTask = Task.Run(() => DatabaseManager.query_db(query_tracking));
        playerData = await playerDataTask;

        // Retrieve frames data asynchronously
        Task<Game[]> frameDataTask = Task.Run(() => DatabaseManager.query_db(query_frames));
        frameData = await frameDataTask;

        Debug.Log("Number of rows: " + playerData.Length);
        Debug.Log("Number of frames: " + frameData.Length);

        CalculateFramesStartAndEndIndex();

        if (playerData != null && frameData != null && frameData.Length > 0)
        {
            // Loop through frames to spawn objects
            for (int i = 0; i < frameData[0].ObjectsTracked; i++)
            {
                SpawnObject(playerData[i]);
            }

            currentFrameNr = 0;
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);

            return true; // Loading successful
        }
        else
        {
            Debug.Log("No frames found");
            return false; // Loading failed
        }
    }

    // Overloaded method to load game using Schedule object
    public async Task<bool> LoadGameAsync(Schedule schedule)
    {
        // Inistialize loading screen
        Debug.Log("Loading game: " + schedule.MatchId);
        gameInfo = schedule;
        // Remove all objects from the scene
        changingGame = true;
        RemoveObjects();
        changingGame = false;
        gameUI.GetComponent<TimeOverlay>().Timer(0);
        homeTeamNameShort.GetComponentInChildren<TMP_Text>().text = schedule.HomeTeamNameShort;
        homeTeamNameShort.GetComponentInChildren<Image>().color = Utils.HexToColor(schedule.HomeTeamColor);
        awayTeamNameShort.GetComponentInChildren<TMP_Text>().text = schedule.AwayTeamNameShort;
        awayTeamNameShort.GetComponentInChildren<Image>().color = Utils.HexToColor(schedule.AwayTeamColor);

        return await LoadGameAsync(schedule.MatchId);
    }

    private void RemoveObjects()
    {
        // Destroy all child objects of home team, away team, and ball
        DestroyChildren(homeTeam);
        DestroyChildren(awayTeam);
        DestroyChildren(ball);
        LineRenderer lineRenderer = GetComponent<LineRenderer>();
        if (lineRenderer != null)
        {
            Destroy(lineRenderer);
        }
        gameObject.GetComponent<GameTools>().DeselectAllTools();
        Debug.Log("Objects removed");

        // Reset the time slider
        timeSlider.GetComponent<TimeSlider>().ChangeTime(startFrame);

        playerData = null;
        frameData = null;
    }

    private void DestroyChildren(GameObject parent)
    {
        // Destroy all child objects of a parent object
        foreach (Transform child in parent.transform)
        {
            Destroy(child.gameObject);
        }
    }


    private void CalculateFramesStartAndEndIndex()
    {
        // Calculate the start and end index of each frame and store in an array
        int frameStartIndex = 0;
        int frameEndIndex = 0;

        int framesLength = frameData.Length;
        framesStartAndEndIndex = new int[framesLength][];

        for (int i = 0; i < framesLength; i++)
        {
            frameStartIndex = frameEndIndex;
            frameEndIndex += frameData[i].ObjectsTracked;
            framesStartAndEndIndex[i] = new int[] { frameStartIndex, frameEndIndex };
        }
    }

    // Get the start index of a frame
    public int GetFrameStartIndex(int frameNr)
    {
        return framesStartAndEndIndex[frameNr][0];
    }

    // Get the end index of a frame
    public int GetFrameEndIndex(int frameNr)
    {
        return framesStartAndEndIndex[frameNr][1];
    }


    private void Play()
    {
        // Play the game by moving objects according to frame data
        if (currentFrameNr < frameData.Length)
        {
            timer += Time.deltaTime;

            if (timer >= interval)
            {
                timer = 0f;

                MoveObjects();

                currentFrameNr++;
                currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
                currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
            }
        }
        else
        {
            Debug.Log("Frame has completed");
        }
    }

    // Toggle play/pause state
    public void PlayPause()
    {
        Debug.Log("PlayPause");
        isPlaying = !isPlaying;
        if (isPlaying)
        {
            playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = pauseIcon;
        }
        else
        {
            playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = playIcon;
        }
    }

    // Set play state to false
    private void SetPlayFalse()
    {
        isPlaying = false;
        playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = playIcon;
    }

    // Fast forward the game
    public void FastForward()
    {
        SetPlayFalse();
        Debug.Log("FastForward");
        if (currentFrameNr + 25 < frameData.Length)
            currentFrameNr += 25;
        else
            currentFrameNr = frameData.Length - 1;

        currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
        currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        MoveObjects();
    }

    // Fast backward the game
    public void FastBackward()
    {
        SetPlayFalse();
        Debug.Log("FastBackward");
        if ((currentFrameNr - 25) > 0)
        {
            Debug.Log("Current frame first if: " + currentFrameNr);
            currentFrameNr -= 25;
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }
        else
        {
            Debug.Log("Current frame second if: " + currentFrameNr);
            currentFrameNr = 0;
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }

        MoveObjects();
    }

    // Step forward the game by one frame
    public void StepForward()
    {
        SetPlayFalse();
        Debug.Log("StepForward");
        if (currentFrameNr + 1 < frameData.Length)
        {
            currentFrameNr++;
            currentFrameEndIndex += frameData[currentFrameNr].ObjectsTracked;
        }
        else
        {
            currentFrameNr = frameData.Length - 1;
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }
        MoveObjects();
    }

    // Step backward the game by one frame
    public void StepBackward()
    {
        SetPlayFalse();
        Debug.Log("StepBackward");
        currentFrameNr--;
        if (currentFrameNr < frameData.Length && currentFrameNr > 0)
        {
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }
        else
        {
            currentFrameNr = 0;
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }
        MoveObjects();
    }

    // Move to a specific frame in the game
    public void MoveTo(int frameNr)
    {
        SetPlayFalse();
        Debug.Log("MoveTo");
        currentFrameNr = frameNr;
        if (currentFrameNr < frameData.Length)
        {
            Debug.Log("Frame found");
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }
        else
        {
            Debug.Log("Frame not found");
            currentFrameStartIndex = GetFrameStartIndex(frameData.Length - 1);
            currentFrameEndIndex = GetFrameEndIndex(frameData.Length - 1);
        }
        MoveObjects();
    }

    // Move objects in the scene according to frame data
    private void MoveObjects()
    {
        Vector3 position;
        Transform playerTransform;

        for (int currentFrameIndex = currentFrameStartIndex; currentFrameIndex < currentFrameEndIndex; currentFrameIndex++)
        {
            string playerName = playerData[currentFrameIndex].Player;
            playerTransform = GameObject.Find(playerName)?.transform;

            if (playerTransform == null)
            {
                Debug.Log("Player not found: " + playerName);
                SpawnObject(playerData[currentFrameIndex]);
            }
            else
            {
                position = playerTransform.position;

                // Move the player to the new position
                position.x = playerData[currentFrameIndex].X;
                position.z = playerData[currentFrameIndex].Y;
                playerTransform.rotation = Quaternion.Euler(0, playerData[currentFrameIndex].Orientation + 90, 0);
                playerTransform.position = position;
            }
        }
        // Update the time slider
        timeSlider.GetComponent<TimeSlider>().ChangeTime(currentFrameNr);
        gameUI.GetComponent<TimeOverlay>().Timer(currentFrameNr);
    }

    // Spawn player or ball object
    private void SpawnObject(Game player)
    {
        Vector3 position;
        if (player.Team == "home_team")
        {
            position = new Vector3(player.X, 0, player.Y);
            SpawnPlayer(position, player, homeTeam, gameInfo.HomeTeamColor);
        }
        else if (player.Team == "away_team")
        {
            position = new Vector3(player.X, 0, player.Y);
            SpawnPlayer(position, player, awayTeam, gameInfo.AwayTeamColor);
        }
        else
        {
            position = new Vector3(player.X, 0.1f, player.Y);
            SpawnBall(position, player);
        }
    }

    // Spawn player object at specified position
    private void SpawnPlayer(Vector3 position, Game player, GameObject team, string teamColor)
    {
        GameObject playerObject = Instantiate(playerPrefab, position, Quaternion.Euler(0, player.Orientation, 0), team.transform) as GameObject;
        playerObject.name = player.Player;
        playerObject.tag = "Player";
        playerObject.GetComponent<Renderer>().material.color = Utils.HexToColor(teamColor);
        playerObject.transform.GetChild(0).gameObject.SetActive(false);
    }

    // Spawn ball object at specified position
    private void SpawnBall(Vector3 position, Game player)
    {
        // Spawn ball object at specified position
        GameObject ballObject = Instantiate(ballPrefab, position, Quaternion.identity, ball.transform) as GameObject;
        ballObject.name = player.Player;
        ballObject.tag = "Player";
        ballObject.transform.GetChild(0).gameObject.SetActive(false);
    }

    // Get the start frame number
    public int StartFrame()
    {
        return startFrame;
    }

    // Get the end frame number
    public int EndFrame()
    {
        return endFrame;
    }
}
