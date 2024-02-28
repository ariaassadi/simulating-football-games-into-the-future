using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;
using UnityEngine.EventSystems;

public class CursorChange : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler
{
    public void OnPointerEnter(PointerEventData eventData)
    {
        Texture2D handPointer = Resources.Load<Texture2D>("pointing_hand");
        // Resize the cursor to 32x32 pixels

        Vector2 hotspot = new Vector2(handPointer.width * 0.4f, handPointer.height / 8); // Center hotspot

        Debug.Log("CursorChange");

        // Change cursor to pointer (hand) when entering the button
        Cursor.SetCursor(handPointer, hotspot, CursorMode.Auto);
    }

    public void OnPointerExit(PointerEventData eventData)
    {
        // Revert cursor to default when exiting the button
        Cursor.SetCursor(null, Vector2.zero, CursorMode.Auto);
    }
}
