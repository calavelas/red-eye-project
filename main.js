document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const inputFile = document.getElementById('image-file');
    const formData = new FormData();
    formData.append('file', inputFile.files[0]);

    let apiUrl = '';
    if (e.submitter.id === 'submit-gif') {
        apiUrl = 'http://localhost:5000/process_image_gif/';
    } else if (e.submitter.id === 'submit-png') {
        apiUrl = 'http://localhost:5000/process_image/';
    } else {
        console.error('Unknown submit button');
        return;
    }

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const blob = await response.blob();
        console.log('Received Blob:', blob); // Debug: Log the received blob
        const imageUrl = URL.createObjectURL(blob);
        console.log('Image URL:', imageUrl); // Debug: Log the created image URL
        const outputImage = document.getElementById('output-image');
        outputImage.src = imageUrl;
        outputImage.style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
    }
});
