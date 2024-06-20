using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using ColorUtils;
public class MenuOptions : MonoBehaviour
{
    // Start is called before the first frame update
    private void Start()
    {
        ThemeManager themeManager = GameObject.Find("UIManager").GetComponent<ThemeManager>();
        Color themeColor = themeManager.GetPrimaryThemeColor();
        ThemeColor.SetThemeColorRecursive(transform, themeColor);
    }
}
