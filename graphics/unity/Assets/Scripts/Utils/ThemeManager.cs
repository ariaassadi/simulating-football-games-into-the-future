using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Utils;
public class ThemeManager : MonoBehaviour
{
    private Color primaryThemeColor;
    private Color secondaryThemeColor;
    // Start is called before the first frame update
    private void Start()
    {
        primaryThemeColor = ColorHelper.HexToColor("#009940");
        secondaryThemeColor = ColorHelper.HexToColor("#FF4B00");
    }

    public Color GetPrimaryThemeColor()
    {
        return primaryThemeColor;
    }

    public Color GetSecondaryThemeColor()
    {
        return secondaryThemeColor;
    }
}
