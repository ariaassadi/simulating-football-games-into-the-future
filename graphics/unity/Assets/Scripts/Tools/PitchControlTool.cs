using UnityEngine;
using System.Collections;
using System.IO;
using System;
using UnityEngine.Networking;

using Utils;
using GameVisualization;

namespace Tools
{
    public class PitchControl : Tool
    {
        private Material squareMaterial; // Material to apply to squares

        private Texture2D texture; // Texture to apply to squares

        private GameObject pitchPrefab; // Plane to apply the texture to

        private GameManager gameManager; // GameManager script

        private GameObject pitch;
        private int rows = 68; // Number of rows in the grid
        private int cols = 105; // Number of columns in the grid

        private string sourcePath;

        protected override void Start()
        {
            base.Start();
            synchronized = true;
            // CreateDatabase();
            sourcePath = Application.streamingAssetsPath + "/Python/pitch_control_socket.py";
            if (Application.platform == RuntimePlatform.Android && !Application.isEditor)
            {
                string destinationPath = Application.persistentDataPath + "/pitch_control_socket.py";
                if (!File.Exists(destinationPath))
                {
                    StartCoroutine(CopyScript(sourcePath, destinationPath));
                }
                sourcePath = destinationPath;
            }
            pitchPrefab = (Resources.Load("SquarePrefab", typeof(GameObject)) as GameObject);
        }

        public override void Select()
        {
            base.Select();
            while (!OpenScript())
            {
                // wait for 3 seconds before trying again
                System.Threading.Thread.Sleep(3000);
            }
            AddPlaneAndTexture();
        }

        public override void Deselect()
        {
            base.Deselect();
            CloseScript();
            RemovePlaneAndTexture();
        }

        private bool OpenScript()
        {
            if (!PythonScript.StartPitchControlScript(sourcePath))
            {
                Debug.LogError("Failed to start pitch control script");
            }
            // wait for 1 second
            System.Threading.Thread.Sleep(1000);
            if (!PythonScript.ConnectToPitchControlScript())
            {
                Debug.LogError("Failed to connect to pitch control script");
                return false;
            }
            return true;
        }

        private void CloseScript()
        {
            if (!PythonScript.CloseConnectionToPitchControlScript())
            {
                Debug.LogError("Failed to close pitch control script");
            }
            if (!PythonScript.StopPitchControlScript())
            {
                Debug.LogError("Failed to stop pitch control script");
            }
        }

        private bool RestartScript()
        {
            CloseScript();
            return OpenScript();
        }

        public override void UpdateTool()
        {
            // Get the GameManager script
            gameManager = GameObject.Find("GameManager").GetComponent<GameManager>();
            // Get playerdata from the GameManager script
            PlayerData[] playerData = gameManager.GetPlayerData();
            UpdatePitchControlTexture(playerData);

            Debug.Log("Updating pitch control texture");
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
            Vector3 position = new Vector3(52.5f, 0.01f, -34f);

            pitch = Instantiate(pitchPrefab, position, Quaternion.identity);
            pitch.tag = "PitchOverlay";

            texture = new Texture2D(cols, rows);

            // Get the GameManager script
            gameManager = GameObject.Find("GameManager").GetComponent<GameManager>();
            // Get playerdata from the GameManager script
            PlayerData[] playerData = gameManager.GetPlayerData();
            UpdatePitchControlTexture(playerData);

        }

        private void AddTransparentTexture()
        {
            texture.filterMode = FilterMode.Point;

            // Generate colors for each pixel based on the data
            for (int x = 0; x < rows; x++)
            {
                for (int y = 0; y < cols; y++)
                {
                    Color color = new Color(0, 0, 0, 0); // Or any method to generate colors based on data
                    texture.SetPixel(y, x, color);
                }
            }

            texture.Apply(); // Apply changes to the texture

            ApplyTexture(texture);
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

            // Free originalTexture from memory:
            Destroy(originalTexture);

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
        private PlayerData MoveOrigin(PlayerData player)
        {
            PlayerData playerCopy = new PlayerData();
            playerCopy.jersey_number = player.jersey_number;
            playerCopy.y = player.y + 34f;
            playerCopy.x = player.x - 52.5f;
            playerCopy.v = player.v;
            playerCopy.orientation = player.orientation;
            playerCopy.team = player.team;
            return playerCopy;
        }
        public void UpdatePitchControlTexture(PlayerData[] playerPositions)
        {
            // move origin to 52.5, 34
            Debug.Log("Players in playerPositions before moving: " + playerPositions.Length);
            PlayerData[] players = new PlayerData[playerPositions.Length];
            for (int i = 0; i < playerPositions.Length; i++)
            {
                players[i] = MoveOrigin(playerPositions[i]);
            }

            if (!BallInPlay(players))
            {
                AddTransparentTexture();
                return;
            }

            // BUG: Too many players, the pitch control script can only handle 23 players

            if (players.Length > 23)
            {
                Debug.Log("Players in playerPositions: " + playerPositions.Length);
                Debug.LogError("Too many players " + players.Length);
                return;
            }
            // Get the JSON to send to the webserver
            string jsonPP = GetPitchControlJSON(players);

            // // Path to store the JSON file
            // string path = Application.temporaryCachePath + "/pitch_control.json";

            // // string path = Application.dataPath + "/Python/pitch_control.json";
            // // Send the JSON to the python script
            // PythonScript.TestPythonScript(jsonPP, path);

            // // Get the JSON and convert it to a float array
            // string jsonPC = System.IO.File.ReadAllText(path);
            // validate jsonPP before sending to python

            if (!JsonParser.ValidatePlayerDataJson(jsonPP))
            {
                Debug.LogError("Invalid player data json");
                return;
            }

            string jsonPC = PythonScript.SendDataToPitchControlScript(jsonPP);

            if (jsonPC == null)
            {
                System.Diagnostics.Stopwatch stopwatch = new System.Diagnostics.Stopwatch();
                stopwatch.Start();
                Debug.LogWarning("Failed to get pitch control data");
                // pause game
                gameManager.PauseGame();
                CloseScript();
                // Restart the script
                while (!OpenScript())
                {
                    // wait for 3 seconds before trying again
                    Debug.LogWarning("Failed to open and connect to script");
                    System.Threading.Thread.Sleep(3000);
                }
                stopwatch.Stop();
                Debug.LogWarning("Time elapsed restarting script: " + stopwatch.Elapsed);
                UpdateTool();
                gameManager.ResumeGame();
            }
            else
            {

                float[,] pitchControlData = JsonParser.ParsePitchJSON(jsonPC);

                // // Generate a new texture
                texture = GenerateTexture(pitchControlData);

                texture = FlipTextureHorizontally(texture);

                // // Apply the new texture
                ApplyTexture(texture);
            }
        }

        private bool BallInPlay(PlayerData[] players)
        {
            foreach (PlayerData player in players)
            {
                if (player.team == "ball")
                {
                    return true;
                }
            }
            return false;
        }

        public void RemovePlaneAndTexture()
        {
            // Remove the plane and texture
            Destroy(texture);
            Destroy(pitch);

            GameObject[] pitchOverlays = GameObject.FindGameObjectsWithTag("PitchOverlay");
            if (pitchOverlays != null && pitchOverlays.Length > 0)
            {
                foreach (GameObject pitchOverlay in pitchOverlays)
                {
                    Destroy(pitchOverlay);
                }
            }
        }

        Texture2D GenerateTexture(float[,] data)
        {
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
            return new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value);
        }

        private Color GenerateColorFromData(Color color, float value)
        {
            // Generate a color based on the data value
            return ColorHelper.ChangeOpacity(color, value);
        }

        private Color GenerateColor(float alpha)
        {
            Color color = new Color();
            if (alpha < 0)
            {
                color = ColorHelper.ChangeOpacity(Color.red, alpha * -1);
            }
            else
            {
                color = ColorHelper.ChangeOpacity(Color.blue, alpha);
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
}