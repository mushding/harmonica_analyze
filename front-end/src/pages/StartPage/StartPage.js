import React from 'react'
import './styles.css'
import Uploader from '../../components/Uploader/Uploader'
import Navbar from '../../components/Navbar/Navbar';
export default class StartPage extends React.Component{
    render(){ 
        return(
            <div>
                <Navbar/>
                <Uploader/>
                <div className="informationContainer">
                   
                </div>
            </div>
        );
    }
}