using System.Collections.Generic;
using UnityEngine;

public class ControlsUI : MonoBehaviour
{
    public GameObject controlsUI;


    void Start()
    {
        Dictionary<string, string> controlsInGame = new Dictionary<string, string>
        {
            { "Move", "WASD" },
            { "Move up", "Space" },
            { "Move Down", "Left Shift" },
            { "Toggle Camera Tilt", "Left Ctrl" },
            { "Jump one second forward ", "l" },
            { "Play/Pause", "k" },
            { "Jump one second backward", "j" },
            { "Jump one frame forward", "."},
            { "Jump one frame backward", ","},
            { "Birds eye view", "b" },
            { "neutral camera position/reset", "n" }
        };

    }
}
