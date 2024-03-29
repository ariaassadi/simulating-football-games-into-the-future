using UnityEngine;
using System.Collections;
using System.IO;
using UnityEngine.Networking;


public class PitchControl : MonoBehaviour
{
    private Material squareMaterial; // Material to apply to squares

    [SerializeField] private GameObject pitchPrefab; // Plane to apply the texture to

    private Component gameManager; // GameManager script

    private float[,] colorBrighness; // Data to generate the texture

    private GameObject pitch;
    private int rows = 68; // Number of rows in the grid
    private int cols = 105; // Number of columns in the grid

    // private Color color = Color.blue; // Color to apply to the texture

    [SerializeField] private TextAsset pitchControlData; // JSON data to generate the texture

    private string json;

    private void Start()
    {
        // CreateDatabase();
        if (Application.platform == RuntimePlatform.Android && !Application.isEditor)
        {
            string sourcePath = Application.streamingAssetsPath + "/Python/pitch_control_main.py";
            string destinationPath = Application.persistentDataPath + "/pitch_control_main.py";
            if (!File.Exists(destinationPath))
                StartCoroutine(CopyScript(sourcePath, destinationPath));
        }


    }

    IEnumerator CopyScript(string sourcePath, string destinationPath)
    {
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
                UnityEngine.Debug.LogError("Failed to copy python scrits at: " + sourcePath + www.error);
            }
        }
    }
    public void AddPlaneAndTexture()
    {
        // Create a new plane and apply a texture to it
        // squareMaterial = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        // squareMaterial.SetFloat("_Surface", 1); // 1 represents transparent
        Vector3 position = new Vector3(52.5f, 0.01f, 34f);
        pitch = Instantiate(pitchPrefab, position, Quaternion.identity);
        pitch.tag = "PitchOverlay";

        // Get the GameManager script
        gameManager = GameObject.Find("GameManager").GetComponent<GameManager>();
        // Get playerdata from the GameManager script
        PlayerData[] playerData = gameManager.GetComponent<GameManager>().GetPlayerData();
        UpdatePitchControlTexture(playerData);
    }

    // Flip texture horizontally
    private Texture2D FlipTextureHorizontally(Texture2D originalTexture)
    {
        Texture2D flippedTexture = new Texture2D(originalTexture.width, originalTexture.height);

        for (int y = 0; y < originalTexture.height; y++)
        {
            for (int x = 0; x < originalTexture.width; x++)
            {
                flippedTexture.SetPixel(originalTexture.width - x - 1, y, originalTexture.GetPixel(x, y));
            }
        }

        flippedTexture.Apply();
        return flippedTexture;
    }

    // Flip texture vertically
    private Texture2D FlipTextureVertically(Texture2D originalTexture)
    {
        Texture2D flippedTexture = new Texture2D(originalTexture.width, originalTexture.height);

        for (int y = 0; y < originalTexture.height; y++)
        {
            for (int x = 0; x < originalTexture.width; x++)
            {
                flippedTexture.SetPixel(x, originalTexture.height - y - 1, originalTexture.GetPixel(x, y));
            }
        }

        flippedTexture.Apply();
        return flippedTexture;
    }

    public void UpdatePitchControlTexture()
    {
        // Generate a new texture
        // Texture2D texture = GenerateTexture();

        // // Apply the new texture
        // ApplyTexture(texture);
    }

    private PlayerData MoveOrigin(PlayerData player)
    {
        PlayerData playerCopy = new PlayerData();
        playerCopy.jersey_number = player.jersey_number;
        playerCopy.y = player.y - 34f;
        playerCopy.x = player.x - 52.5f;
        playerCopy.v = player.v;
        playerCopy.orientation = player.orientation;
        playerCopy.team = player.team;
        return playerCopy;
    }
    public void UpdatePitchControlTexture(PlayerData[] playerPositions)
    {

        // move origin to 52.5, 34
        PlayerData[] players = new PlayerData[playerPositions.Length];
        for (int i = 0; i < playerPositions.Length; i++)
        {
            players[i] = MoveOrigin(playerPositions[i]);
        }

        // Get the JSON to send to the webserver
        string jsonPP = GetPitchControlJSON(players);

        // Path to store the JSON file
        string path = Application.temporaryCachePath + "/pitch_control.json";

        // string path = Application.dataPath + "/Python/pitch_control.json";
        // Send the JSON to the python script
        PythonScript.TestPythonScript(jsonPP, path);



        // Get the JSON and convert it to a float array
        string jsonPC = System.IO.File.ReadAllText(path);

        float[,] pitchControlData = JsonParser.ParsePitchJSON(jsonPC);

        // // Generate a new texture
        Texture2D texture = GenerateTexture(pitchControlData);

        texture = FlipTextureHorizontally(texture);

        // // Apply the new texture
        ApplyTexture(texture);
    }

    public void RemovePlaneAndTexture()
    {
        // Remove the plane and texture
        Destroy(pitch);
    }

    // Texture2D GenerateTexture()
    // {
    //     Texture2D texture = new Texture2D(cols, rows);

    //     string path = Application.temporaryCachePath + "/output.json";

    //     UnityEngine.Debug.Log(path);

    //     // read the json file from path

    //     // Time this function
    //     Stopwatch stopwatch = new Stopwatch();
    //     stopwatch.Start();

    //     PythonScript.TestPythonScript();

    //     stopwatch.Stop();
    //     UnityEngine.Debug.Log("Time taken to load write JSON: " + stopwatch.ElapsedMilliseconds + "ms");

    //     stopwatch = new Stopwatch();
    //     stopwatch.Start();

    //     // TextAsset jsonFile = Resources.Load<TextAsset>("output");
    //     string json = System.IO.File.ReadAllText(path);

    //     stopwatch.Stop();
    //     UnityEngine.Debug.Log("Time taken to read the JSON: " + stopwatch.ElapsedMilliseconds + "ms");

    //     if (json == null)
    //     {
    //         UnityEngine.Debug.Log("hohohohohohoh not found");
    //         return null;
    //     }

    //     // Retrieve color brightness data from JsonParser
    //     stopwatch = new Stopwatch();
    //     stopwatch.Start();
    //     PitchData pitchData = JsonUtility.FromJson<PitchData>(json);
    //     stopwatch.Stop();
    //     UnityEngine.Debug.Log("Time taken to parse JSON: " + stopwatch.ElapsedMilliseconds + "ms");

    //     float[] hmmmm = pitchData.pitch;
    //     foreach (float value in hmmmm)
    //     {
    //         UnityEngine.Debug.Log(value);
    //     }
    //     texture.filterMode = FilterMode.Point;

    //     // Generate colors for each pixel
    //     for (int x = 0; x < rows; x++)
    //     {
    //         for (int y = 0; y < cols; y++)
    //         {
    //             Color color = GenerateRandomColor(); // Or any method to generate colors based on data
    //             texture.SetPixel(y, x, color);
    //         }
    //     }

    //     texture.Apply(); // Apply changes to the texture

    //     return texture;
    // }
    // Texture2D GenerateTexture()
    // {
    //     Texture2D texture = new Texture2D(cols, rows);

    //     // Time this function
    //     Stopwatch stopwatch = new Stopwatch();
    //     stopwatch.Start();

    //     TextAsset jsonFile = Resources.Load<TextAsset>("pitch_control");

    //     // end time
    //     stopwatch.Stop();
    //     UnityEngine.Debug.Log("Time taken to load JSON into text asset: " + stopwatch.ElapsedMilliseconds + "ms");

    //     if (jsonFile == null)
    //     {
    //         UnityEngine.Debug.Log("File not found");
    //         return null;
    //     }

    //     // Retrieve color brightness data from JsonParser
    //     stopwatch = new Stopwatch();
    //     stopwatch.Start();
    //     colorBrighness = JsonParser.ParsePitchJSON(jsonFile.text);
    //     stopwatch.Stop();
    //     UnityEngine.Debug.Log("Time taken to parse JSON: " + stopwatch.ElapsedMilliseconds + "ms");

    //     texture.filterMode = FilterMode.Point;

    //     // get rows and cols from the data
    //     rows = colorBrighness.GetLength(0);
    //     cols = colorBrighness.GetLength(1);
    //     UnityEngine.Debug.Log("Rows: " + rows + " Cols: " + cols);
    //     // Generate colors for each pixel
    //     for (int x = 0; x < rows; x++)
    //     {
    //         for (int y = 0; y < cols; y++)
    //         {
    //             Color color = GenerateColor(colorBrighness[x, y]); // Or any method to generate colors based on data
    //             texture.SetPixel(y, x, color);
    //         }
    //     }

    //     texture.Apply(); // Apply changes to the texture

    //     return texture;
    // }

    Texture2D GenerateTexture(float[,] data)
    {
        Texture2D texture = new Texture2D(cols, rows);
        texture.filterMode = FilterMode.Point;

        // Generate colors for each pixel based on the data
        for (int x = 0; x < rows; x++)
        {
            for (int y = 0; y < cols; y++)
            {
                Color color = GenerateColor(data[x, y]); // Or any method to generate colors based on data
                texture.SetPixel(y, x, color);
            }
        }

        texture.Apply(); // Apply changes to the texture

        return texture;
    }

    private void ApplyTexture(Texture2D texture)
    {
        // Get material from the plane
        squareMaterial = pitch.GetComponent<Renderer>().material;
        // Apply the texture to the material

        squareMaterial.mainTexture = texture;
        pitch.GetComponent<Renderer>().material = squareMaterial;
        // // Apply the material to each square GameObject
        // GameObject[] squares = GameObject.FindGameObjectsWithTag("PitchOverlay"); // Assuming squares have a "Square" tag
        // foreach (GameObject square in squares)
        // {
        //     Renderer renderer = square.GetComponent<Renderer>();
        //     renderer.sharedMaterial = squareMaterial;
        // }
    }

    private Color GenerateRandomColor()
    {
        // Generate a random color
        return new Color(Random.value, Random.value, Random.value);
    }

    private Color GenerateColorFromData(Color color, float value)
    {
        // Generate a color based on the data value
        return Utils.ChangeOpacity(color, value);
    }

    private Color GenerateColor(float alpha)
    {
        Color color = new Color();
        if (alpha < 0)
        {
            color = Utils.ChangeOpacity(Color.red, alpha * -1);
        }
        else
        {
            color = Utils.ChangeOpacity(Color.blue, alpha);
        }
        // Color color = Utils.GenerateColorGradient(alpha, Color.blue, Color.red);
        // Color color = Utils.GenerateColorGradient(alpha, Utils.HexToColor(GameObject.Find("GameManager").GetComponent<GameManager>().GetHomeTeamColor()), Utils.HexToColor(GameObject.Find("GameManager").GetComponent<GameManager>().GetAwayTeamColor()));
        return color;
    }

    private float[,] GetColors(Vector2[] playerPositions)
    {
        // Get colors for each data point from Thiago's script
        float[,] colors = new float[cols, rows];
        return colors;
    }

    // The JSON to send to webserver to get the pitch control data
    private string GetPitchControlJSON(PlayerData[] playerData)
    {
        // Get the JSON to send to the webserver
        string json = JsonParser.PlayerDataArrayToJson(playerData);
        return json;
    }
}