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
    public class ColorHelper : MonoBehaviour
    {
        /// <summary>
        /// Convert hex color to Color
        /// </summary>
        /// <param name="hex"></param>
        /// <returns>The Color</returns>
        public static Color HexToColor(string hex)
        {
            if (hex.StartsWith("#"))
            {
                hex = hex.Substring(1);
            }
            if (hex.Length != 6)
            {
                Debug.LogError("Invalid hex color");
                return Color.white;
            }
            float r = int.Parse(hex.Substring(0, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
            float g = int.Parse(hex.Substring(2, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
            float b = int.Parse(hex.Substring(4, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
            return new Color(r, g, b);
        }

        /// <summary>
        /// Change opacity of a color using hex
        /// </summary>
        /// <param name="hex"></param>
        /// <param name="alpha"></param>
        /// <returns>The Color</returns>
        public static Color HexToColor(string hex, float alpha)
        {
            Color color = HexToColor(hex);
            color.a = alpha;
            return color;
        }

        public static Color ChangeOpacity(Color color, float alpha)
        {
            color.a = alpha;
            return color;
        }

        public static Color ChangeBrightness(Color color, float factor)
        {
            float h, s, v;
            Color.RGBToHSV(color, out h, out s, out v);
            v *= factor;
            return Color.HSVToRGB(h, s, v);
        }

        public static Color GenerateColorGradient(float value)
        {
            // Clamp the value between -1 and 1
            value = Mathf.Clamp(value, -1f, 1f);

            // Define the colors for red, white, and blue
            Color red = Color.red;
            Color white = Color.white;
            Color blue = Color.blue;

            // Interpolate between the colors based on the input value
            if (value < 0f)
            {
                return Color.Lerp(red, white, Mathf.InverseLerp(-1f, 0f, value));
            }
            else
            {
                return Color.Lerp(white, blue, Mathf.InverseLerp(0f, 1f, value));
            }
        }

        // GEnerateColor Gradient with custom colors
        public static Color GenerateColorGradient(float value, Color homeColor, Color awayColor)
        {
            // Clamp the value between -1 and 1
            value = Mathf.Clamp(value, -1f, 1f);

            // Interpolate between the colors based on the input value
            if (value < 0f)
            {
                return Color.Lerp(awayColor, Color.white, Mathf.InverseLerp(-1f, 0f, value));
            }
            else
            {
                return Color.Lerp(Color.white, homeColor, Mathf.InverseLerp(0f, 1f, value));
            }
        }
    }

}

// [System.Serializable]
// public class Coordinates
// {
//     public float x;
//     public float y;
//     public override string ToString()
//     {
//         return $"x={x.ToString()}, z={y.ToString()}";
//     }
// }


// public class JSONParser
// {
//     [SerializeField] public TextAsset file;

//     // public static Coordinates[] ParseJSONFile(string filePath)
//     // {
//     //     using (StreamReader streamReader = new StreamReader(filePath))
//     //     {
//     //         string jsonString = streamReader.ReadToEnd();
//     //         // Debug.Log(jsonString);
//     //         Coordinates[] positions = JsonConvert.DeserializeObject<Coordinates>(jsonString);
//     //         // Debug.Log("PLAYA: " + matchData.players[0].player_name);
//     //         // Debug.Log("PLAYA CO x: " + matchData.players[0].coordinates[0][0].ToString());
//     //         // Debug.Log("PLAYA CO: " + matchData.players[0].coordinates.ToString());

//     //         return positions;
//     //     }
//     // }

//     public static Coordinates[] ParseJSONFile(string filePath)
//     {
//         using (StreamReader streamReader = new StreamReader(filePath))
//         {
//             string jsonString = streamReader.ReadToEnd();
//             Debug.Log(jsonString);
//             Coordinates[] positions = JsonHelper.FromJson<Coordinates>(jsonString);
//             // Debug.Log("PLAYA: " + matchData.players[0].player_name);
//             // Debug.Log("PLAYA CO x: " + matchData.players[0].coordinates[0][0].ToString());
//             // Debug.Log("PLAYA CO: " + matchData.players[0].coordinates.ToString());
//             if (positions == null)
//             {
//                 Debug.Log("No positions found");
//                 return null;
//             }
//             foreach (Coordinates pos in positions)
//             {
//                 Debug.Log("POS: " + pos.ToString());
//             }
//             return positions;
//         }
//     }
// }

// public static class JsonHelper
// {
//     public static T[] FromJson<T>(string json)
//     {
//         Wrapper<T> wrapper = JsonUtility.FromJson<Wrapper<T>>(json);
//         return wrapper.Items;
//     }

//     public static string ToJson<T>(T[] array)
//     {
//         Wrapper<T> wrapper = new Wrapper<T>();
//         wrapper.Items = array;
//         return JsonUtility.ToJson(wrapper);
//     }

//     public static string ToJson<T>(T[] array, bool prettyPrint)
//     {
//         Wrapper<T> wrapper = new Wrapper<T>();
//         wrapper.Items = array;
//         return JsonUtility.ToJson(wrapper, prettyPrint);
//     }

//     [System.Serializable]
//     private class Wrapper<T>
//     {
//         public T[] Items;
//     }
// }