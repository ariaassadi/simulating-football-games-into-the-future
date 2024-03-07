using UnityEngine;
using System.Data;
using System;
using Unity.VisualScripting;

public class CreateAndMovePlayers : MonoBehaviour
{
    [SerializeField] private GameObject playerHomePrefab;
    [SerializeField] private GameObject playerAwayPrefab;
    [SerializeField] private GameObject ballPrefab;

    [SerializeField] private GameObject homeTeam;
    [SerializeField] private GameObject awayTeam;
    [SerializeField] private GameObject ball;

    [SerializeField] private Texture2D playIcon;
    [SerializeField] private Texture2D pauseIcon;

    [SerializeField] private GameObject playPauseButton;

    [SerializeField] private GameObject timeSlider;


    private TimeOverlay timeOverlay;

    [SerializeField] private int startFrame = 0;
    [SerializeField] private int endFrame = 67500;

    private int currentFrameStartIndex;
    private int currentFrameEndIndex;
    private int currentFrameNr;

    private Schedule[] schedule;

    private Game[] playerData;
    private Game[] frameData;

    float timer = 0f;
    float interval = 0.04f; // 40 milliseconds in seconds
    private bool isPlaying = false;

    private int[][] framesStartAndEndIndex;

    void Start()
    {
        timeOverlay = GetComponent<TimeOverlay>();

        schedule = DatabaseManager.query_schedule_db($"SELECT match_id FROM game_schedule");

        if (schedule != null)
        {
            string query = $"SELECT player, x, y, frame, team_direction, orientation FROM games_table_orientation WHERE frame>={startFrame} AND frame<{endFrame} AND period=1 AND match_id='{schedule[0].MatchId}'";

            Debug.Log($"Select: {query} END");
            // Retrieve players data
            playerData = DatabaseManager.query_db(query);

            // Retrieve frames data
            frameData = DatabaseManager.query_db($"SELECT frame, objects_tracked FROM games_table_orientation WHERE frame>={startFrame} AND frame<{endFrame} AND period=1 GROUP BY frame");

            Debug.Log(playerData.Length);
            Debug.Log(frameData.Length);

            CalculateFramesStartAndEndIndex();

            if (playerData != null && frameData != null && frameData.Length > 0)
            {
                // Loop through frames
                for (int i = 0; i < frameData[0].ObjectsTracked; i++)
                {
                    SpawnObject(playerData[i]);
                }

                currentFrameNr = 0;
                currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
                currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
            }
            else
            {
                Debug.Log("No frames found");
            }
        }
    }

    private void CalculateFramesStartAndEndIndex()
    {
        int frameStartIndex = 0;
        int frameEndIndex = 0;

        // Initialize the array to store the start and end index of each frame
        // has size of the number of frames
        int framesLength = frameData.Length;
        framesStartAndEndIndex = new int[framesLength][];

        // Loop through eanch frame and store how many objects are in each frame
        for (int i = 0; i < framesLength; i++)
        {
            frameStartIndex = frameEndIndex;
            frameEndIndex += frameData[i].ObjectsTracked;
            framesStartAndEndIndex[i] = new int[] { frameStartIndex, frameEndIndex };
        }
    }

    public int GetFrameStartIndex(int frameNr)
    {
        return framesStartAndEndIndex[frameNr][0];
    }

    public int GetFrameEndIndex(int frameNr)
    {
        return framesStartAndEndIndex[frameNr][1];
    }

    void Update()
    {
        if (isPlaying)
        {
            Play();
        }
    }

    private void Play()
    {
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

    private void SetPlayFalse()
    {
        isPlaying = false;
        playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = playIcon;
    }

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

        // Debug.Log("Current frame start: " + currentFrameStartIndex);
        // Debug.Log("Current frame end: " + currentFrameEndIndex);
        MoveObjects();
    }
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

    void MoveObjects()
    {
        Vector3 position;
        Transform playerTransform;

        // Debug.Log("Current frame: " + currentFrameStartIndex);

        for (int currentFrameIndex = currentFrameStartIndex; currentFrameIndex < currentFrameEndIndex; currentFrameIndex++)
        {
            string playerName = playerData[currentFrameIndex].Player;
            playerTransform = GameObject.Find(playerName)?.transform;

            if (playerTransform == null)
            {
                Debug.Log("Player not found: " + playerName);
                if (playerData[currentFrameIndex].TeamDirection == "right")
                {
                    position = new Vector3(playerData[currentFrameIndex].X, 0, playerData[currentFrameIndex].Y);
                    SpawnPlayer(position, playerData[currentFrameIndex], playerHomePrefab, homeTeam);
                }
                else
                {
                    position = new Vector3(playerData[currentFrameIndex].X, 0, playerData[currentFrameIndex].Y);
                    SpawnPlayer(position, playerData[currentFrameIndex], playerAwayPrefab, awayTeam);
                }
                playerTransform = GameObject.Find(playerName)?.transform;
            }
            else
            {
                // Cache the player's transform for reuse
                position = playerTransform.position;

                // Move the player to the new position
                // maybe adda check for the ball

                position.x = playerData[currentFrameIndex].X;
                position.z = playerData[currentFrameIndex].Y;
                playerTransform.rotation = Quaternion.Euler(0, playerData[currentFrameIndex].Orientation + 90, 0);
            }
            playerTransform.position = position;
        }
        // Update the time slider
        timeSlider.GetComponent<TimeSlider>().ChangeTime(currentFrameNr);
        timeOverlay.Timer(currentFrameNr);
    }


    void SpawnObject(Game player)
    {
        Vector3 position;
        if (player.TeamDirection == "right")
        {
            position = new Vector3(105 - player.X, 0, 68 - player.Y);
            SpawnPlayer(position, player, playerHomePrefab, homeTeam);
        }
        else if (player.TeamDirection == "left")
        {
            position = new Vector3(player.X, 0, player.Y);
            SpawnPlayer(position, player, playerAwayPrefab, awayTeam);
        }
        else
        {
            position = new Vector3(player.X, 0.1f, player.Y);
            SpawnBall(position, player);
        }
    }
    void SpawnPlayer(Vector3 position, Game player, GameObject playerPrefab, GameObject team)
    {
        GameObject playerObject = Instantiate(playerPrefab, position, Quaternion.Euler(0, player.Orientation, 0), team.transform) as GameObject;
        playerObject.name = player.Player;
    }

    void SpawnBall(Vector3 position, Game player)
    {
        GameObject ballObject = Instantiate(ballPrefab, position, Quaternion.identity, ball.transform) as GameObject;
        ballObject.name = player.Player;
    }

    public int StartFrame()
    {
        return startFrame;
    }
    public int EndFrame()
    {
        return endFrame;
    }
}
