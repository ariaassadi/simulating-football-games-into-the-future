using UnityEngine;


public class ShowPlayerNamesTool : Tool
{
    protected override void Start()
    {
        base.Start();
    }

    public override void Select()
    {
        base.Select();
        players = GameObject.FindGameObjectsWithTag("Player");
        ShowPlayerNames();
        Debug.Log("Show player names tool selected");
    }

    public override void Deselect()
    {
        base.Deselect();
        HidePlayerNames();
        Debug.Log("Show player names tool deselected");
    }

    public override void UpdateTool()
    {
        return;
    }

    private void ShowPlayerNames()
    {
        if (players != null)
        {
            for (int i = 0; i < players.Length; i++)
            {
                print("Player: " + players[i].name);
                ShowPlayerName(players[i]);
            }
        }
    }

    private void HidePlayerNames()
    {
        if (players != null)
        {
            foreach (GameObject player in players)
            {
                if (player != null)
                {
                    HidePlayerName(player);

                }
            }
        }
    }
}