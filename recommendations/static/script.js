document.addEventListener("DOMContentLoaded", () => {
    // load models and actions
    prepareModelsChoice()
})

const destroyMessageBoxes = (parent = document.body) => {
    let boxes = parent.querySelectorAll('.messageBox')
    boxes.forEach(box => {
            box.remove()
    });
}

const createMessageBox = (message, type = 'error', parent = document.body) => {
    let messageClass = 'error'
    switch (type) {
        case 'success':
            messageClass = 'success'
            break;
        case 'info':
            messageClass = 'info'
            break;
        case 'warning':
            messageClass = 'warning'
            break;
        default:
            break;
    }
    const box = document.createElement("div")
    box.classList.add('messageBox', messageClass)
    box.innerHTML = message
    parent.appendChild(box)
    setTimeout(()=>{destroyMessageBoxes(parent)}, 5000);
}

const prepareModelsChoice = () => {
    const modelSelection = document.querySelector("#model_selection")
    modelSelection.addEventListener('change', event => {
        const selectedOption = event.target.value
        const data = {
            model: selectedOption
        }
        fetch('./load_rbm_model',{
            method: 'POST', 
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        }).then(response => {
            if (response.ok) {
                return response.json()
            } else {
                throw new Error('Error: ' + response.statusText)
            }
        }).then(data => {
            if(data.success==true){
                // wczytany model, wyświetlenie akcji
                createMessageBox(data.message, type='success')
                fetch('./actions')
                .then(response => {
                    if (response.ok) {
                        return response.text()
                    } else {
                        throw new Error('Error: ' + response.statusText)
                    }
                }).then(data => {
                    document.getElementById('content').innerHTML = data
                })
            }
            else{
                createMessageBox('Błąd wczytywania modelu')
                throw new Error('Error: ' + data.message)
            }
        })
        
    })
}

const newUser = () => {
    fetch('/initial', {
        method: 'POST', 
        headers: {
            'Content-Type': 'application/json',
        }
    }).then(response => {
        if (response.ok) {
            return response.text()
        } else {
            throw new Error('Error: ' + response.statusText)
        }
    }).then(data => {
        createMessageBox('Poprawnie otwarto proces zbierania danych', type='success')
        document.getElementById('content').innerHTML = data
        // Submit 
        document.getElementById('movies_choice').addEventListener('submit', e => {
            e.preventDefault()
            const formData = new FormData()
            const checkboxes = document.querySelectorAll('input[name="chosen_movies"]:checked')
            checkboxes.forEach(checkbox => {
                formData.append('chosen_movies', checkbox.value)
            })
            fetch('/new_user_recommendations', {
                method: 'POST',
                body: formData
            })
            .then(response => response.text())
            .then(data => document.getElementById('content').innerHTML = data)
            .catch(error => console.error('Error:', error))
        });
    })
}

function nextPage() {
    const recommendations = document.querySelectorAll('.recommendations');
    for (let i = 0; i < recommendations.length; i++) {
        if (recommendations[i].classList.contains('active')) {
            recommendations[i].classList.remove('active')
            if (i === recommendations.length - 1) {
                recommendations[0].classList.add('active')
            } else {
                recommendations[i + 1].classList.add('active')
            }
            break
        }
    }
}
function prevPage() {
    const recommendations = document.querySelectorAll('.recommendations');
    for (let i = 0; i < recommendations.length; i++) {
        if (recommendations[i].classList.contains('active')) {
            recommendations[i].classList.remove('active')
            if (i === 0) {
                recommendations[recommendations.length - 1].classList.add('active')
            } else {
                recommendations[i - 1].classList.add('active')
            }
            break
        }
    }
}

const existingUser = () => {

}