document.addEventListener("DOMContentLoaded", () => {
    // load models and actions
    prepareModelsChoice()
})

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
                throw new Error('Error: ' + data.message)
            }
        })
        
    })
}

const newUser = () => {

}

const existingUser = () => {

}