using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace ColorUtils
{
    public class ThemeColor : MonoBehaviour
    {
        public static void SetThemeColor(Transform parent, Color themeColor)
        {
            parent.GetComponent<UnityEngine.UI.Image>().color = themeColor;
        }

        public static void SetThemeColorChild(Transform parent, Color themeColor)
        {
            foreach (Transform child in parent)
            {
                if (child.GetComponent<UnityEngine.UI.Image>())
                {
                    child.GetComponent<UnityEngine.UI.Image>().color = themeColor;
                }
            }
        }

        public static void SetThemeColorRecursive(Transform parent, Color themeColor)
        {
            foreach (Transform child in parent)
            {
                if (child.GetComponent<UnityEngine.UI.Image>())
                {
                    child.GetComponent<UnityEngine.UI.Image>().color = themeColor;
                }

                // Check if the current child has children, and recursively call the function
                if (child.childCount > 0)
                {
                    SetThemeColorRecursive(child, themeColor);
                }
            }
        }
    }
}
