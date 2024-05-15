using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Linq;

namespace Utils
{
    public static class JsonParser
    {
        ////////////////////////
        /// Pitch Data
        ////////////////////////

        public static void WriteJsonToFile(string json, string path)
        {
            Debug.Log("Writing JSON " + json + "\nto " + path);
            File.WriteAllText(path, json);
        }

        ////////////////////////
        /// Pitch Data
        ////////////////////////

        // Convert a flattend pitch array in JSON to a 2D array of floats
        public static float[,] ParsePitchJSON(string json)
        {
            // Read jsonString from a file
            // string jsonString = File.ReadAllText("Assets/Static/Coordinates/pitch_control.json");

            // Parse JSON into a wrapper class
            PitchData pitchData = JsonUtility.FromJson<PitchData>(json);

            // Access the pitch values
            int rows = 68;
            int cols = 105;
            if (pitchData.pitch.Length != rows * cols)
            {
                Debug.LogError("Invalid pitch data");
                return null;
            }
            // convert pitchData to a 2D array of floats
            float[,] pitch = new float[rows, cols];
            int index = 0;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    pitch[i, j] = pitchData.pitch[index];
                    index++;
                }
            }

            return pitch;
        }

        ////////////////////////
        /// Player Data
        ////////////////////////


        // Convert an array of PlayerData to JSON
        public static string PlayerDataArrayToJson(PlayerData[] playerDataArray)
        {
            PlayerDataWrapper wrapper = new PlayerDataWrapper();
            wrapper.players = playerDataArray;
            string jsonString = JsonUtility.ToJson(wrapper);
            // Debug.Log("Converted to JSON: " + jsonString);
            return jsonString;
        }

        public static bool ValidatePlayerDataJson(string jsonData)
        {
            try
            {
                // Deserialize the JSON string into a JObject
                JObject jsonObj = JObject.Parse(jsonData);

                // Check if the JSON object contains the "players" array
                if (jsonObj["players"] == null)
                {
                    UnityEngine.Debug.LogError("Missing 'players' array in JSON data.");
                    return false;
                }

                // Deserialize the "players" array into an array of PlayerData objects
                PlayerDataWrapper playerDataWrapper = JsonConvert.DeserializeObject<PlayerDataWrapper>(jsonData);

                // Check if the deserialization was successful
                if (playerDataWrapper == null || playerDataWrapper.players == null)
                {
                    UnityEngine.Debug.LogError("Failed to deserialize JSON data into PlayerDataWrapper object.");
                    return false;
                }

                // Check if the "players" array contains valid PlayerData objects
                foreach (PlayerData playerData in playerDataWrapper.players)
                {
                    if (playerData == null)
                    {
                        UnityEngine.Debug.LogError("Invalid player data object in the 'players' array.");
                        return false;
                    }
                }

                UnityEngine.Debug.Log("JSON data validation successful.");
                return true;
            }
            catch (JsonReaderException ex)
            {
                UnityEngine.Debug.LogError($"Error parsing JSON data: {ex.Message}");
                return false;
            }
            catch (Exception ex)
            {
                UnityEngine.Debug.LogError($"Error validating JSON data: {ex.Message}");
                return false;
            }
        }

        ////////////////////////
        /// Controls
        ////////////////////////

        public static Controls[] GetControlsFromJson(string json)
        {
            ControlsWrapper controlsWrapper = JsonUtility.FromJson<ControlsWrapper>(json);
            return controlsWrapper.controls;
        }

        ////////////////////////
        /// Settings
        ////////////////////////

        public static Settings GetSettingsFromJson(string json)
        {
            Settings settings = JsonUtility.FromJson<Settings>(json);
            return settings;
        }

        public static string GetJsonFromSettings(Settings settings)
        {
            string jsonString = JsonUtility.ToJson(settings);
            return jsonString;
        }


        ////////////////////////
        /// Game Data
        ////////////////////////

        public static Game[] GetGameFromJson(string json)
        {
            GameWrapper gameWrapper = JsonUtility.FromJson<GameWrapper>(json);
            Game[] games = gameWrapper.game;
            Debug.Log("Game lenght: " + games.Length);
            Debug.Log("First game object: " + games[0].player);
            return games;
        }

    }
}

[System.Serializable]
public class Game
{
    public string team;
    public string team_name;
    public string team_direction;
    public int jersey_number;
    public string player;
    public float x;
    public float y;
    public int frame;
    public int period;
    public int objects_tracked;
    public float x_future;
    public float y_future;
    public float v_x;
    public float v_y;
    public float v_x_avg;
    public float v_y_avg;
    public float orientation;
    public float distance_to_ball;
    public float angle_to_ball;
    public string nationality;
    public float height;
    public int weight;
    public int acc;
    public int pac;
    public int sta;
    public string position;
    public float tiredness;
    public float tiredness_short;
    public string match_id;

    public override string ToString()
    {
        return $"Player: {this.player}, Frame: {this.frame}, Team: {this.team}, Jersey Number: {this.jersey_number}, Position: ({this.x}, {this.y}), Future Position: ({this.x_future}, {this.y_future})";
    }
}

[System.Serializable]
public class GameWrapper
{
    public Game[] game;
}

[System.Serializable]
public class PitchData
{
    public float[] pitch;
}

[System.Serializable]
public class PlayerData
{
    public float x;
    public float y;
    public string team;
    public int jersey_number;
    public string player_name;
    public float v;
    public float orientation;

    public float x_future;
    public float y_future;

    public float offside;

    public override string ToString()
    {
        return $"Player: {player_name}, Team: {team}, Jersey Number: {jersey_number}, Position: ({x}, {y}), Future Position: ({x_future}, {y_future})";
    }
}

[System.Serializable]
public class PlayerDataWrapper
{
    public PlayerData[] players;
}

[System.Serializable]
public class Controls
{
    public string action;
    public string key;

    public override string ToString()
    {
        return $"Action: {action}, Key: {key}";
    }
}

[System.Serializable]
public class ControlsWrapper
{
    public Controls[] controls;
}

[System.Serializable]
public class Settings
{
    public float rotationSpeed;
    public float horizontalSpeed;
    public float verticalSpeed;

    public override string ToString()
    {
        return $"Rotation Speed: {rotationSpeed}, Horizontal Speed: {horizontalSpeed}, Vertical Speed: {verticalSpeed}";
    }
}