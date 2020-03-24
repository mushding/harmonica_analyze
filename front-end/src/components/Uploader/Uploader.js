import React from 'react';
import './styles.css'
import Collapse from '@material-ui/core/Collapse';
import ColorCircularProgress from '../ProgressCircle/ProgressCircle'
import ShowWave from '../ShowWave/ShowWave'
class Uploader extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            checked: false,
            isUploadFile: false,
            filename: "default",
            progressState: "idle",
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
        // handle read data from form
        ev.preventDefault();
        const data = new FormData();
        
        // handle file max size
        if (this.uploadInput.files.length){
            const maxSize = 1024 * 1024 * 10
            let fileSize = this.uploadInput.files[0].size
            if (fileSize > maxSize){
                alert("file size is more then " + 10 + " MBs.")
                return false
            }
        } else {
            alert('Please select a wav file first.')
            return false
        }
        data.append('file', this.uploadInput.files[0]);
        this.setState({ filename: this.uploadInput.files[0].name })
        
        // handle progress circle
        if (this.state.progressState !== 'idle') {
            this.setState({ progressState: 'idle' });
            return;
        }
        this.setState({ progressState: 'progress' });
        

        fetch('https://www.haranalyzer.site/upload', {
            method: 'POST',
            body: data,
        }).then((response) => {
            if(response.ok){
                this.setState({ progressState: 'success' })
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
    toggleSelect = (event) => {
        event.persist()
        this.setState({filename: event.target.value});
    }
    toggleSelectButton = () => {
        if (this.state.filename === "default") {
            alert("Please select a example.")
            return false
        }
        this.setState({ isUploadFile: true })
    }
    render() {
        if (!this.state.isUploadFile){
            return(
                <div>
                    <Collapse in={this.state.checked} timeout={2500}>
                        <div className="uploadContainer">
                            <form onSubmit={this.handleUploadImage} className="formContainer">
                                <div className="fromPadding">
                                    <label htmlFor="file-upload" className="custom-file-upload">
                                        Select wav file HERE
                                    </label>
                                    <input ref={(ref) => { this.uploadInput = ref; }} type="file" id="file-upload" accept="audio/wav"/>
                                </div>
                                <div className="fromPadding">
                                    <button disabled={this.state.progressState === 'progress'}>Upload</button>
                                </div>
                            </form>
                            <div className="formContainer">
                                <div className="field">
									<label htmlFor="demo-category">Or Select examples</label>
									<select value={this.state.filename} name="demo-category" id="demo-category" onChange={this.toggleSelect}>
										<option value="default">-</option>
										<option value="little_star.wav">Little Stars</option>
									</select>
								</div>
                                <div className="fromPadding">
                                    <button onClick={this.toggleSelectButton} className="selectButton" disabled={this.state.progressState === 'progress'}>Upload</button>
                                </div>
                            </div>
                            {this.state.progressState === 'success' ? (
                                <div className="hidden"></div>
                            ) : (
                                <Collapse
                                    in={this.state.progressState === 'progress'}
                                    timeout={2000}
                                    unmountOnExit
                                >
                                    <div className="progressContainer">
                                        <ColorCircularProgress />
                                    </div>
                                </Collapse>
                            )}
                        </div>
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