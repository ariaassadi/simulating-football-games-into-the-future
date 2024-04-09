using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ToolManager : MonoBehaviour
{
    private Tool currentTool;

    public void SelectTool(GameObject toolObject)
    {
        Tool tool = toolObject.GetComponent<Tool>();
        if (tool != null)
        {
            if (currentTool != null)
            {
                currentTool.Deselect();
            }
            if (currentTool == tool)
            {
                currentTool.Deselect();
                currentTool = null;
            }
            else
            {
                currentTool = tool;
                currentTool.Select();
            }
        }
    }

    // public void Update()
    // {
    //     if (currentTool != null)
    //         currentTool.Update();
    // }

}

// public abstract class Tool : MonoBehaviour
// {
//     private GameObject[] players;
//     public abstract void Select();
//     public abstract void Deselect();
//     public abstract void Update();
// }

// public class CalculateDistance : Tool
// {
//     public override void Select()
//     {
//         Debug.Log("Distance tool selected");
//     }

//     public override void Deselect()
//     {
//         Debug.Log("Distance tool deselected");
//     }

//     public override void Update()
//     {
//         Debug.Log("Calculating distance");
//     }
// }
