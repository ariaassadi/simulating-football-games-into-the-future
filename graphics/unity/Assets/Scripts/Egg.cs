using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace Eggs
{
    public class Egg : MonoBehaviour
    {
        private bool isDragging = false;

        private void Update()
        {
            if (isDragging)
            {
                MoveEgg();
            }
        }
        private void MoveEgg()
        {
            if (!isDragging)
            {
                Debug.Log("Egg is not dragging");
                return;
            }

            RaycastHit hit;
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out hit))
            {
                if (hit.transform.gameObject.tag == "Pitch")
                {
                    Vector3 position = hit.point;
                    position.y = 0;

                    gameObject.transform.position = position;
                }
            }
        }

        public void SetIsDragging(bool isDragging)
        {
            this.isDragging = isDragging;
        }
    }
}

