using UnityEngine;
using UnityEngine.UI;

public class Utils : MonoBehaviour
{
    public static Color HexToColor(string hex)
    {
        Debug.Log(hex);
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
}