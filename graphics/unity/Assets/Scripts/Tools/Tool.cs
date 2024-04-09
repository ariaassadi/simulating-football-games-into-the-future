using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public abstract class Tool : MonoBehaviour
{
    // private GameObject[] players;
    protected Image border;

    private void Start()
    {
        border = transform.Find("Border").GetComponent<Image>();
        if (border == null)
        {
            Debug.LogError("Border not found");
            return;
        }
        else
        {
            Debug.Log("Border found for " + this.name);
        }
    }

    public abstract void Select();
    public abstract void Deselect();
    public abstract void Update();
}