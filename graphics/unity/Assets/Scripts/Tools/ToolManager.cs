using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ToolManager : MonoBehaviour
{
    private Tool[] activeTools;
    private GameObject[] players;

    public void Update()
    {
        if (activeTools != null)
        {
            foreach (Tool tool in activeTools)
            {
                tool.UpdateTool();
            }
        }
    }

    public void SelectTool(GameObject toolObject)
    {
        Tool tool = toolObject.GetComponent<Tool>();
        if (tool != null)
        {
            if (tool.uniqueTool)
            {
                if (ToolInActiveTools(tool))
                {
                    DeselectTool(tool);
                }
                else
                {
                    DeselectTools();
                    SelectToolHelper(tool);
                }
            }
            else if (ToolInActiveTools(tool))
            {
                DeselectTool(tool);
            }
            else
            {
                SelectToolHelper(tool);
                if (players == null)
                    Debug.Log("Number of players in manager select: " + 0);
                else
                    Debug.Log("Number of players in manager select: " + players.Length);

            }
        }
    }

    private void AddToSelectedTools(Tool tool)
    {
        if (activeTools == null)
        {
            activeTools = new Tool[1];
            activeTools[0] = tool;
        }
        else
        {
            Tool[] newActiveTools = new Tool[activeTools.Length + 1];
            for (int i = 0; i < activeTools.Length; i++)
            {
                newActiveTools[i] = activeTools[i];
            }
            newActiveTools[activeTools.Length] = tool;
            activeTools = newActiveTools;
        }
    }

    private bool ToolInActiveTools(Tool tool)
    {
        if (activeTools != null)
        {
            foreach (Tool activeTool in activeTools)
            {
                if (activeTool == tool)
                {
                    return true;
                }
            }
        }
        return false;
    }

    // If any tool that has property uniqueToo set to true, deselect that tool
    private void SelectToolHelper(Tool tool)
    {
        if (activeTools != null)
        {
            foreach (Tool activeTool in activeTools)
            {
                if (activeTool.uniqueTool)
                {
                    DeselectTool(activeTool);
                }
            }
        }
        tool.Select();
        AddToSelectedTools(tool);
    }

    private void DeselectTool(Tool tool)
    {
        if (activeTools != null)
        {
            Tool[] newActiveTools = new Tool[activeTools.Length - 1];
            int j = 0;
            for (int i = 0; i < activeTools.Length; i++)
            {
                if (activeTools[i] != tool)
                {
                    newActiveTools[j] = activeTools[i];
                    j++;
                }
                else
                {
                    activeTools[i].Deselect();
                }
            }
            activeTools = newActiveTools;
        }
    }
    private void DeselectTools()
    {
        if (activeTools != null)
        {
            foreach (Tool tool in activeTools)
            {
                tool.Deselect();
            }
            activeTools = null;
        }
    }

    public void SetPlayers(GameObject[] players)
    {
        this.players = players;
    }

}
