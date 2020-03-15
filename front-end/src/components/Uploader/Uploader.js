import React from 'react';
import './styles.css'
import Collapse from '@material-ui/core/Collapse';
import ShowWave from '../../components/ShowWave/ShowWave'

class Uploader extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            checked: false,
            isUploadFile: false,
            filename: "",
        }
    }
    componentDidMount() {
        var that = this;
		setTimeout(() => {
			that.show();
		}, that.props.wait);
    }
    show = () => {
        var that = this
		var delay = (s) => {
            return new Promise((resolve, reject) => {
                setTimeout(resolve,s); 
            });
        };
        delay().then(() => {
            delay(500)
            that.setState({ checked: true })
        })
    }
    handleUploadImage = (ev) => {
        ev.preventDefault();

        const data = new FormData();
        data.append('file', this.uploadInput.files[0]);
        this.setState({ filename: this.uploadInput.files[0].name })
        console.log(this.state.filename)
        // console.log(this.uploadInput.files[0].name)
        // data.append('filename', this.fileName.value);

        fetch('http://192.168.50.225:5000/upload', {
            method: 'POST',
            body: data,
        }).then((response) => {
            if(response.ok){
                return (response)
            } else {
                throw new Error('Something went wrong');
            }
        }).then(() => {
            console.log("IN!!!!")
            this.setState({ isUploadFile: true })
        }).catch((error) => {
            console.log(error)
        })
    }
    render() {
        if (!this.state.isUploadFile){
            return(
                <div>
                    <Collapse in={this.state.checked} timeout={2500}>
                        <form onSubmit={this.handleUploadImage} className="formContainer">
                            <div>
                                <input ref={(ref) => { this.uploadInput = ref; }} type="file" />
                            </div>
                            <br />
                            <div>
                                <button>Upload</button>
                            </div>
                        </form>
                    </Collapse>
                </div>     
            );
        }
        else {
            return(
                <div>
                    <ShowWave file={this.state.filename}/>
                </div>
            );
        }
    }
}

export default Uploader;